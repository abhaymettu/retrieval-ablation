"""Two passes.

Pass one dumps the queries worth reading: the ones the recommended stack still
fails, and the ones where BM25 beats the dense retriever. That file is read by a
human, who writes the labels into analysis/failure_labels.json by hand.

Pass two joins those hand labels back onto the data and produces the counts and
the worked examples the README quotes. No labels file means pass two is skipped
rather than faked.
"""
import gzip
import json

from config import CACHE, DATASETS, OUTPUT, ROOT
from lib import (apply_rerank, load_corpus, load_json, load_run, load_split,
                 ndcg_at_k, ranked_ids, save_json, system_list,
                 best_system_on_dev)

LABELS = ROOT / "analysis" / "failure_labels.json"
DUMP_PER_BUCKET = 40


def system_run(dataset, split, name, tuning):
    for sysname, first_stage, k in system_list(tuning):
        if sysname == name:
            base = load_run(dataset, split, first_stage)
            if not k:
                return base
            with gzip.open(CACHE / f"{dataset}.{split}.rerank.json.gz", "rt") as fh:
                return apply_rerank(base, json.load(fh), k)
    raise KeyError(name)


def candidates(dataset, tuning, results):
    corpus = load_corpus(dataset)
    queries, qrels = load_split(dataset, "test")
    best = best_system_on_dev(results, dataset)

    runs = {
        "best": system_run(dataset, "test", best, tuning),
        "bm25": load_run(dataset, "test", "bm25"),
        "dense": load_run(dataset, "test", f"dense-{tuning['best_dense']}"),
    }

    rows = []
    for qid in sorted(qrels):
        n = {k: ndcg_at_k(ranked_ids(r.get(qid, [])), qrels[qid], 10)
             for k, r in runs.items()}
        rows.append({
            "qid": qid, "query": queries[qid],
            "ndcg_best": round(n["best"], 4),
            "ndcg_bm25": round(n["bm25"], 4),
            "ndcg_dense": round(n["dense"], 4),
            "bm25_minus_dense": round(n["bm25"] - n["dense"], 4),
            "n_positives": sum(1 for r in qrels[qid].values() if r > 0),
            "gold": [{"doc": d, "rel": qrels[qid][d], "text": corpus[d][:400]}
                     for d in sorted(qrels[qid], key=lambda x: -qrels[qid][x])[:3]
                     if qrels[qid][d] > 0],
            "top3_best": [{"doc": d, "rel": qrels[qid].get(d, 0), "text": corpus[d][:400]}
                          for d in ranked_ids(runs["best"].get(qid, []))[:3]],
        })

    fails = [r for r in rows if r["ndcg_best"] == 0.0]
    bm25_wins = sorted([r for r in rows if r["bm25_minus_dense"] > 0],
                       key=lambda r: -r["bm25_minus_dense"])
    return {
        "dataset": dataset, "best_system": best,
        "total_queries": len(rows),
        "total_zero_ndcg": len(fails),
        "total_bm25_beats_dense": len(bm25_wins),
        "zero_ndcg": fails[:DUMP_PER_BUCKET],
        "bm25_beats_dense": bm25_wins[:DUMP_PER_BUCKET],
        "all_rows": rows,
    }


def summarise(dumps):
    """Join hand labels onto the dumped queries."""
    labels = load_json(LABELS)
    out = {"modes": labels["modes"], "datasets": {}}
    for ds, dump in dumps.items():
        assigned = labels["labels"].get(ds, {})
        by_qid = {r["qid"]: r for r in dump["all_rows"]}
        unknown = [q for q in assigned if q not in by_qid]
        assert not unknown, f"{ds}: labelled queries not in test split: {unknown[:3]}"

        counts, examples = {}, {}
        for qid, mode in assigned.items():
            assert mode in labels["modes"], f"{ds}/{qid}: unknown mode {mode}"
            counts[mode] = counts.get(mode, 0) + 1
            r = by_qid[qid]
            examples.setdefault(mode, []).append({
                "qid": qid, "query": r["query"],
                "ndcg_best": r["ndcg_best"], "ndcg_bm25": r["ndcg_bm25"],
                "ndcg_dense": r["ndcg_dense"],
                "gold_snippet": r["gold"][0]["text"][:220] if r["gold"] else "",
                "top1_snippet": r["top3_best"][0]["text"][:220] if r["top3_best"] else "",
            })
        out["datasets"][ds] = {
            "best_system": dump["best_system"],
            "labelled": len(assigned),
            "total_zero_ndcg": dump["total_zero_ndcg"],
            "total_queries": dump["total_queries"],
            "counts": counts,
            "share": {m: round(100 * c / len(assigned), 1) for m, c in counts.items()},
            "examples": examples,
        }
    return out


if __name__ == "__main__":
    tunings = {t["dataset"]: t for t in load_json(OUTPUT / "tuning.json")}
    results = load_json(OUTPUT / "results.json")

    dumps = {ds: candidates(ds, tunings[ds], results) for ds in DATASETS}
    readable = {ds: {k: v for k, v in d.items() if k != "all_rows"}
                for ds, d in dumps.items()}
    save_json(readable, OUTPUT / "error_candidates.json")
    for ds, d in dumps.items():
        print(f"{ds}: best on dev is {d['best_system']}, "
              f"{d['total_zero_ndcg']}/{d['total_queries']} queries score nDCG@10 = 0, "
              f"{d['total_bm25_beats_dense']} where BM25 beats dense")

    if LABELS.exists():
        save_json(summarise(dumps), OUTPUT / "failure_modes.json")
        print(f"wrote {OUTPUT / 'failure_modes.json'}")
    else:
        print(f"\nno {LABELS}, skipping the taxonomy. "
              f"Read output/error_candidates.json and write the labels by hand.")
