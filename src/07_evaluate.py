"""Score every system on every metric, with bootstrap intervals on each mean
and a paired bootstrap on every pairwise difference.

Reads only cached runs and cached reranker scores, so it is offline and cheap.
"""
import gzip
import itertools
import json

from config import (BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, CACHE, DATASETS,
                    OUTPUT, RERANK_DEPTH)
from lib import (METRICS, apply_rerank, load_json, load_run, load_split,
                 mean_ci, paired_bootstrap, per_query_scores, save_json,
                 system_list)


def rerank_scores(dataset, split):
    with gzip.open(CACHE / f"{dataset}.{split}.rerank.json.gz", "rt") as fh:
        return json.load(fh)


def build_runs(dataset, split, tuning):
    ce = rerank_scores(dataset, split)
    runs = {}
    for name, first_stage, k in system_list(tuning):
        base = load_run(dataset, split, first_stage)
        runs[name] = apply_rerank(base, ce, k) if k else base
    return runs


def evaluate(dataset, split, tuning):
    _, qrels = load_split(dataset, split)
    runs = build_runs(dataset, split, tuning)

    scores = {m: {} for m in METRICS}
    systems = {}
    for name, run in runs.items():
        systems[name] = {}
        for metric in METRICS:
            _, arr = per_query_scores(run, qrels, metric)
            scores[metric][name] = arr
            mean, lo, hi = mean_ci(arr, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
            systems[name][metric] = {"mean": round(mean, 4),
                                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}

    comparisons = []
    for metric in METRICS:
        for a, b in itertools.combinations(runs, 2):
            diff, lo, hi, p = paired_bootstrap(scores[metric][a], scores[metric][b],
                                               BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
            comparisons.append({
                "metric": metric, "a": a, "b": b,
                "diff": round(diff, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "p": round(p, 4),
                "distinguishable": bool(lo > 0 or hi < 0),
            })

    # What the reranker could possibly recover: it can only reorder what the
    # first stage returned, so first stage Recall@100 is a hard ceiling on any
    # k = 100 reranked system.
    ceilings = {}
    for name, first_stage, k in system_list(tuning):
        if k == RERANK_DEPTH:
            ceilings[name] = systems[first_stage]["recall@100"]["mean"]

    # The other ceiling: with only 100 slots, a query with more than 100
    # relevant documents cannot reach Recall@100 even with perfect retrieval.
    # This separates "the cutoff is too small" from "the retriever missed them",
    # which are very different problems with very different fixes.
    per_query_cap = [min(100, p) / p for p in
                     (sum(1 for r in v.values() if r > 0) for v in qrels.values()) if p]
    label_cap = float(sum(per_query_cap) / len(per_query_cap))

    return {"dataset": dataset, "split": split, "queries": len(qrels),
            "systems": systems, "comparisons": comparisons,
            "recall_ceiling_at_k100": ceilings,
            "recall100_cap_from_cutoff": round(label_cap, 4),
            "queries_with_over_100_positives": sum(
                1 for v in qrels.values()
                if sum(1 for r in v.values() if r > 0) > 100)}


if __name__ == "__main__":
    tunings = {t["dataset"]: t for t in load_json(OUTPUT / "tuning.json")}
    out = {}
    for ds in DATASETS:
        out[ds] = {split: evaluate(ds, split, tunings[ds]) for split in ["dev", "test"]}
        t = out[ds]["test"]["systems"]
        print(f"\n{ds} (test, {out[ds]['test']['queries']} queries)")
        for name, m in t.items():
            n = m["ndcg@10"]
            print(f"  {name:22s} nDCG@10 {n['mean']:.4f} "
                  f"[{n['ci_lo']:.4f}, {n['ci_hi']:.4f}]  "
                  f"R@100 {m['recall@100']['mean']:.4f}")
    save_json(out, OUTPUT / "results.json")
