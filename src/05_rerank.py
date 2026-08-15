"""Cross encoder scores for the top RERANK_DEPTH of every candidate first stage.

Pairs are deduplicated across first stages before scoring, so asking what the
reranker does on top of BM25 as well as on top of hybrid costs far less than
twice one pass. Scores are cached, so k = 10, 50 and 100 all read the same file.
"""
import gzip
import json
import time

import torch
from sentence_transformers import CrossEncoder

from config import CACHE, DATASETS, EMBED_MODELS, OUTPUT, RERANK_DEPTH, RERANKER
from lib import load_corpus, load_run, load_split, save_json

FIRST_STAGES = ["bm25", "hybrid-rrf", "hybrid-blend"] + [f"dense-{m}" for m in EMBED_MODELS]


def cache_path(dataset, split):
    return CACHE / f"{dataset}.{split}.rerank.json.gz"


def run(dataset, split, model, stats):
    out = cache_path(dataset, split)
    if out.exists():
        print(f"{dataset} {split}: cached")
        return

    corpus = load_corpus(dataset)
    queries, _ = load_split(dataset, split)

    pairs, keys = [], []
    seen = set()
    for system in FIRST_STAGES:
        for qid, ranked in load_run(dataset, split, system).items():
            for doc, _ in ranked[:RERANK_DEPTH]:
                if (qid, doc) not in seen:
                    seen.add((qid, doc))
                    keys.append((qid, doc))
                    pairs.append((queries[qid], corpus[doc]))

    t0 = time.perf_counter()
    scores = model.predict(pairs, batch_size=128, show_progress_bar=True)
    elapsed = time.perf_counter() - t0

    scored = {}
    for (qid, doc), s in zip(keys, scores):
        scored.setdefault(qid, {})[doc] = float(s)
    with gzip.open(out, "wt") as fh:
        json.dump(scored, fh)

    stats.append({"dataset": dataset, "split": split, "pairs": len(pairs),
                  "seconds": round(elapsed, 1),
                  "pairs_per_s": round(len(pairs) / elapsed, 1)})
    print(f"{dataset} {split}: {len(pairs)} pairs in {elapsed:.1f}s")


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CrossEncoder(RERANKER["hf"], device=device, max_length=RERANKER["max_seq"])
    stats = []
    for ds in DATASETS:
        for split in ["dev", "test"]:
            run(ds, split, model, stats)
    if stats:
        save_json(stats, OUTPUT / "rerank_throughput.json")
