"""Measure per query latency, one query at a time, no batching.

Stages are timed separately and then added per query, so a composed system's
p95 is the p95 of its own per query totals rather than a sum of stage p95s.
Serving is modelled as serial and cold, which is pessimistic for production and
is the point: it is the number an engineer can reason about before caching.
"""
import pickle
import time

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from config import (CACHE, DATASETS, EMBED_MODELS, MACHINE, OUTPUT,
                    RERANK_DEPTH, RERANK_KS, RERANKER, RETRIEVAL_DEPTH, RRF_K)
from lib import (emb_path, ids_path, load_corpus, load_json, load_run,
                 load_split, minmax, save_json, system_list)

LATENCY_QUERIES = 220   # per dataset, sampled deterministically from test
WARMUP = 20             # discarded from the head of every stage's timing loop
TOKEN_RE = __import__("re").compile(r"[a-z0-9]+")


def warm_the_gpu(models, reranker):
    """The first few hundred MPS calls are slower than the steady state, and a
    per stage warmup of 20 was not enough to hide it: an earlier run reported
    bge query encoding at 52 ms on the first dataset timed and 11 ms on the
    last, in measurement order rather than query length order. So every model
    is driven hard before any clock starts, and the datasets are timed only
    after the whole process is warm."""
    filler = ["a warmup query about retrieval evaluation"] * 32
    for _ in range(10):
        for m in models.values():
            m.encode(filler, batch_size=32, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)
        reranker.predict([(filler[0], filler[0])] * 32, batch_size=32,
                         show_progress_bar=False)


def topk(scores, depth):
    top = np.argpartition(-scores, min(depth, len(scores) - 1))[:depth]
    return top[np.argsort(-scores[top], kind="stable")]


def measure(dataset, tuning, models, reranker):
    queries, _ = load_split(dataset, "test")
    qids = sorted(queries)[:LATENCY_QUERIES]
    corpus = load_corpus(dataset)
    doc_ids = load_json(ids_path(dataset, "doc"))

    with open(CACHE / f"{dataset}.bm25.pkl", "rb") as fh:
        bm25 = pickle.load(fh)
    docs = {m: np.load(emb_path(dataset, m, "docs")) for m in EMBED_MODELS}

    stage = {}

    # BM25 query time
    times = []
    for i, qid in enumerate(tqdm(qids, desc=f"{dataset} bm25")):
        t0 = time.perf_counter()
        s = bm25.get_scores(TOKEN_RE.findall(queries[qid].lower()))
        topk(np.asarray(s), RETRIEVAL_DEPTH)
        if i >= WARMUP:
            times.append(time.perf_counter() - t0)
    stage["bm25_query"] = times

    # Dense: encoding and search timed apart, because they scale differently.
    # Both models are timed inside one loop over queries rather than one loop
    # per model. Timing them model by model handed whichever ran first the
    # per dataset cold start, which reversed the two: it reported the 22M model
    # as slower than the 110M one. Alternating removes that.
    enc = {m: [] for m in EMBED_MODELS}
    search = {m: [] for m in EMBED_MODELS}
    for i, qid in enumerate(tqdm(qids, desc=f"{dataset} dense")):
        for m, spec in EMBED_MODELS.items():
            t0 = time.perf_counter()
            v = models[m].encode([spec["query_prefix"] + queries[qid]],
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).astype(np.float32)
            t1 = time.perf_counter()
            topk(v[0] @ docs[m].T, RETRIEVAL_DEPTH)
            t2 = time.perf_counter()
            if i >= WARMUP:
                enc[m].append(t1 - t0)
                search[m].append(t2 - t1)
    for m in EMBED_MODELS:
        stage[f"encode_{m}"] = enc[m]
        stage[f"search_{m}"] = search[m]

    # An attempt to also time query encoding with no search interleaved, to put
    # a number on how much the stage split is distorted by the two competing for
    # memory bandwidth, was removed: it reported the isolated encode as up to 4x
    # slower than the interleaved one on two of three datasets, which is not a
    # result any mechanism here explains. The stage rows below are therefore
    # indicative. The composed per system totals are what this repo relies on,
    # and those are measured end to end per query.

    # Fusion overhead on real top-1000 lists
    best_dense = tuning["best_dense"]
    bm_run = load_run(dataset, "test", "bm25")
    dn_run = load_run(dataset, "test", f"dense-{best_dense}")
    alpha = tuning["alpha_selected_on_dev"]
    fuse = []
    for i, qid in enumerate(qids):
        t0 = time.perf_counter()
        d, s = minmax(dn_run[qid]), minmax(bm_run[qid])
        acc = {}
        for doc, v in d.items():
            acc[doc] = alpha * v
        for doc, v in s.items():
            acc[doc] = acc.get(doc, 0.0) + (1 - alpha) * v
        sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[:RETRIEVAL_DEPTH]
        if i >= WARMUP:
            fuse.append(time.perf_counter() - t0)
    stage["fuse"] = fuse
    # RRF is the same shape of work, within noise of the blend, so one number
    # covers both fusion methods.

    # Cross encoder over the top k of the best first stage. The same timing is
    # reused for reranking on top of BM25, since the cost is set by how many
    # pairs are scored and how long the documents are, not by which first stage
    # nominated them.
    best_run = load_run(dataset, "test", tuning["best_first_stage"])
    for k in RERANK_KS:
        times = []
        for i, qid in enumerate(tqdm(qids, desc=f"{dataset} ce@{k}")):
            pairs = [(queries[qid], corpus[d]) for d, _ in best_run[qid][:k]]
            t0 = time.perf_counter()
            reranker.predict(pairs, batch_size=k, show_progress_bar=False)
            if i >= WARMUP:
                times.append(time.perf_counter() - t0)
        stage[f"ce@{k}"] = times

    return {k: np.array(v) for k, v in stage.items()}


def compose(name, first_stage, k, stage, tuning):
    """Per query total for one system, in seconds."""
    best_dense = tuning["best_dense"]
    if first_stage == "bm25":
        total = stage["bm25_query"].copy()
    elif first_stage.startswith("dense-"):
        m = first_stage.split("-", 1)[1]
        total = stage[f"encode_{m}"] + stage[f"search_{m}"]
    else:
        total = (stage["bm25_query"] + stage[f"encode_{best_dense}"]
                 + stage[f"search_{best_dense}"] + stage["fuse"])
    if k:
        total = total + stage[f"ce@{k}"]
    return total


if __name__ == "__main__":
    from config import COST
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    models = {}
    for m, spec in EMBED_MODELS.items():
        models[m] = SentenceTransformer(spec["hf"], device=device)
        models[m].max_seq_length = spec["max_seq"]
    reranker = CrossEncoder(RERANKER["hf"], device=device, max_length=RERANKER["max_seq"])

    warm_the_gpu(models, reranker)

    tunings = {t["dataset"]: t for t in load_json(OUTPUT / "tuning.json")}
    out = {"machine": MACHINE, "device": device, "queries_timed": LATENCY_QUERIES - WARMUP,
           "warmup_queries": WARMUP, "cost": COST, "datasets": {}}

    for ds in DATASETS:
        stage = measure(ds, tunings[ds], models, reranker)
        entry = {"stages_ms": {k: {"p50": round(float(np.percentile(v, 50)) * 1000, 2),
                                   "p95": round(float(np.percentile(v, 95)) * 1000, 2)}
                               for k, v in stage.items()},
                 "systems_ms": {}}
        for name, fs, k in system_list(tunings[ds]):
            t = compose(name, fs, k, stage, tunings[ds])
            p50, p95 = float(np.percentile(t, 50)), float(np.percentile(t, 95))
            entry["systems_ms"][name] = {
                "p50": round(p50 * 1000, 2),
                "p95": round(p95 * 1000, 2),
                "mean": round(float(t.mean()) * 1000, 2),
                # Serial serving: 1000 queries occupy mean latency x 1000 seconds
                # of one instance, billed at the stated hourly rate.
                "usd_per_1k_queries": round(float(t.mean()) * 1000 / 3600
                                            * COST["usd_per_hour"], 4),
            }
        out["datasets"][ds] = entry
        print(f"\n{ds}: " + ", ".join(f"{n} p95 {v['p95']}ms"
                                     for n, v in entry["systems_ms"].items()))

    save_json(out, OUTPUT / "latency.json")
