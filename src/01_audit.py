"""Audit the datasets before building anything on them. Corpus size, query
counts, label density, relevance grades, empty documents, and whether dev and
test queries overlap."""
import numpy as np

from config import DATASETS, OUTPUT
from lib import load_corpus, load_split, save_json, split_name


def audit(dataset):
    corpus = load_corpus(dataset)
    lengths = np.array([len(t.split()) for t in corpus.values()])
    empty = sum(1 for t in corpus.values() if not t)

    row = {
        "dataset": dataset,
        "corpus_docs": len(corpus),
        "empty_docs": empty,
        "doc_words_median": float(np.median(lengths)),
        "doc_words_p95": float(np.percentile(lengths, 95)),
        "splits": {},
    }

    seen = {}
    for split in ["dev", "test"]:
        queries, qrels = load_split(dataset, split)
        judged = [len(v) for v in qrels.values()]
        positives = [sum(1 for r in v.values() if r > 0) for v in qrels.values()]
        grades = {}
        for v in qrels.values():
            for r in v.values():
                grades[str(r)] = grades.get(str(r), 0) + 1
        missing_docs = sum(1 for v in qrels.values() for d in v if d not in corpus)
        row["splits"][split] = {
            "official_split": split_name(dataset, split),
            "queries": len(queries),
            "queries_with_qrels": len(qrels),
            "judged_per_query_mean": round(float(np.mean(judged)), 2),
            "positives_per_query_mean": round(float(np.mean(positives)), 2),
            "queries_with_no_positive": sum(1 for p in positives if p == 0),
            "label_density_pct": round(100 * sum(judged) / (len(qrels) * len(corpus)), 6),
            "grades": grades,
            "qrel_docs_missing_from_corpus": missing_docs,
            "query_words_median": float(np.median([len(q.split()) for q in queries.values()])),
        }
        seen[split] = set(qrels)

    overlap = seen["dev"] & seen["test"]
    row["dev_test_query_overlap"] = len(overlap)
    assert not overlap, f"{dataset}: dev and test share {len(overlap)} query ids"
    return row


if __name__ == "__main__":
    rows = [audit(d) for d in DATASETS]
    save_json(rows, OUTPUT / "audit.json")
    for r in rows:
        print(f"\n{r['dataset']}: {r['corpus_docs']} docs, "
              f"median {r['doc_words_median']:.0f} words, {r['empty_docs']} empty")
        for split, s in r["splits"].items():
            print(f"  {split} (official '{s['official_split']}'): {s['queries']} queries, "
                  f"{s['positives_per_query_mean']} positives/query, "
                  f"grades {s['grades']}, "
                  f"density {s['label_density_pct']}%")
    print(f"\nwrote {OUTPUT / 'audit.json'}")
