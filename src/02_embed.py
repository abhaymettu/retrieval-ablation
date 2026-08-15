"""Embed every corpus and every query once, cache to disk. Everything
downstream reads these files, so the expensive part never runs twice.

Also records index build time and on disk index size per dataset and model.
"""
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import CACHE, DATASETS, EMBED_MODELS, OUTPUT
from lib import emb_path, ids_path, load_corpus, load_split, save_json


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def encode(model, texts, batch_size):
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,       # cosine becomes a dot product
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)


def build(dataset, model_key, stats):
    spec = EMBED_MODELS[model_key]
    corpus = load_corpus(dataset)
    doc_ids = sorted(corpus)
    save_json(doc_ids, ids_path(dataset, "doc"))

    query_ids, query_texts = [], []
    for split in ["dev", "test"]:
        queries, _ = load_split(dataset, split)
        ids = sorted(queries)
        save_json(ids, ids_path(dataset, f"query_{split}"))
        query_ids += ids
        query_texts += [spec["query_prefix"] + queries[q] for q in ids]

    doc_out = emb_path(dataset, model_key, "docs")
    if doc_out.exists():
        print(f"{dataset}/{model_key}: cached")
        return

    model = SentenceTransformer(spec["hf"], device=device())
    model.max_seq_length = spec["max_seq"]

    t0 = time.perf_counter()
    doc_vecs = encode(model, [spec["doc_prefix"] + corpus[d] for d in doc_ids], 64)
    build_s = time.perf_counter() - t0

    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(doc_out, doc_vecs)

    offset = 0
    for split in ["dev", "test"]:
        n = len(load_split(dataset, split)[0])
        q_vecs = encode(model, query_texts[offset:offset + n], 64)
        np.save(emb_path(dataset, model_key, f"queries_{split}"), q_vecs)
        offset += n

    stats.append({
        "dataset": dataset,
        "model": model_key,
        "hf_model": spec["hf"],
        "params_m": spec["params_m"],
        "dim": spec["dim"],
        "docs": len(doc_ids),
        "index_build_s": round(build_s, 1),
        "docs_per_s": round(len(doc_ids) / build_s, 1),
        "index_mb": round(doc_out.stat().st_size / 1e6, 1),
        "device": device(),
    })
    print(f"{dataset}/{model_key}: {len(doc_ids)} docs in {build_s:.1f}s, "
          f"{doc_out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    path = OUTPUT / "index_stats.json"
    stats = []
    if path.exists():
        from lib import load_json
        stats = load_json(path)
    for ds in DATASETS:
        for mk in EMBED_MODELS:
            build(ds, mk, stats)
    save_json(stats, path)
