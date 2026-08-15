"""First stage runs: BM25 and both dense bi-encoders, dev and test.

BM25 uses the k1 and b that BEIR's own Anserini baseline publishes (0.9, 0.4).
They are not tuned here, on either split.
"""
import pickle
import re
import time

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from config import CACHE, DATASETS, EMBED_MODELS, OUTPUT, RETRIEVAL_DEPTH
from lib import (emb_path, ids_path, load_corpus, load_json, load_split,
                 save_json, save_run)

BM25_K1, BM25_B = 0.9, 0.4
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN.findall(text.lower())


def bm25(dataset, stats):
    corpus = load_corpus(dataset)
    doc_ids = load_json(ids_path(dataset, "doc"))

    t0 = time.perf_counter()
    index = BM25Okapi([tokenize(corpus[d]) for d in doc_ids], k1=BM25_K1, b=BM25_B)
    build_s = time.perf_counter() - t0

    CACHE.mkdir(parents=True, exist_ok=True)
    blob = CACHE / f"{dataset}.bm25.pkl"
    with open(blob, "wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)

    stats.append({
        "dataset": dataset, "model": "bm25", "hf_model": f"rank_bm25 Okapi k1={BM25_K1} b={BM25_B}",
        "params_m": 0.0, "dim": 0, "docs": len(doc_ids),
        "index_build_s": round(build_s, 1),
        "docs_per_s": round(len(doc_ids) / build_s, 1),
        "index_mb": round(blob.stat().st_size / 1e6, 1), "device": "cpu",
    })

    ids = np.array(doc_ids)
    for split in ["dev", "test"]:
        queries, _ = load_split(dataset, split)
        run = {}
        for qid in tqdm(sorted(queries), desc=f"bm25 {dataset} {split}"):
            scores = index.get_scores(tokenize(queries[qid]))
            top = np.argpartition(-scores, min(RETRIEVAL_DEPTH, len(scores) - 1))[:RETRIEVAL_DEPTH]
            top = top[np.argsort(-scores[top], kind="stable")]
            run[qid] = [[str(ids[i]), float(scores[i])] for i in top]
        save_run(run, dataset, split, "bm25")
    print(f"bm25 {dataset}: built in {build_s:.1f}s, {blob.stat().st_size / 1e6:.1f} MB")


def dense(dataset, model_key):
    doc_ids = np.array(load_json(ids_path(dataset, "doc")))
    docs = np.load(emb_path(dataset, model_key, "docs"))
    for split in ["dev", "test"]:
        qids = load_json(ids_path(dataset, f"query_{split}"))
        qvecs = np.load(emb_path(dataset, model_key, f"queries_{split}"))
        run = {}
        # Chunked so a large corpus never materialises a full score matrix.
        for start in range(0, len(qids), 64):
            sims = qvecs[start:start + 64] @ docs.T
            for row, qid in enumerate(qids[start:start + 64]):
                s = sims[row]
                top = np.argpartition(-s, min(RETRIEVAL_DEPTH, len(s) - 1))[:RETRIEVAL_DEPTH]
                top = top[np.argsort(-s[top], kind="stable")]
                run[qid] = [[str(doc_ids[i]), float(s[i])] for i in top]
        save_run(run, dataset, split, f"dense-{model_key}")
        print(f"dense-{model_key} {dataset} {split}: {len(run)} queries")


if __name__ == "__main__":
    path = OUTPUT / "index_stats.json"
    stats = [s for s in load_json(path) if s["model"] != "bm25"] if path.exists() else []
    for ds in DATASETS:
        bm25(ds, stats)
        for mk in EMBED_MODELS:
            dense(ds, mk)
    save_json(stats, path)
