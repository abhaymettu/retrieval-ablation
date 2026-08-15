"""Loading, metrics, fusion and statistics. Everything numeric lives here so
tests.py can check it against hand computed values."""
import gzip
import json
import math
from collections import defaultdict

import numpy as np

from config import CACHE, DATA, DEV_SPLIT, RUNS


# ---------------------------------------------------------------- BEIR loading

def load_corpus(dataset):
    """doc_id -> text. Title and body joined the way BEIR's own loader does."""
    corpus = {}
    with open(DATA / dataset / "corpus.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            title = (d.get("title") or "").strip()
            text = (d.get("text") or "").strip()
            corpus[d["_id"]] = (title + " " + text).strip()
    return corpus


def load_queries(dataset):
    queries = {}
    with open(DATA / dataset / "queries.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            queries[d["_id"]] = d["text"].strip()
    return queries


def load_qrels(dataset, split):
    """query_id -> {doc_id: relevance}. Official BEIR qrels, unmodified."""
    qrels = defaultdict(dict)
    path = DATA / dataset / "qrels" / f"{split}.tsv"
    with open(path) as fh:
        header = next(fh)
        assert header.split()[:3] == ["query-id", "corpus-id", "score"], header
        for line in fh:
            qid, did, score = line.rstrip("\n").split("\t")[:3]
            qrels[qid][did] = int(score)
    return dict(qrels)


def split_name(dataset, split):
    """'dev' resolves to whichever official split this dataset lends for tuning."""
    return DEV_SPLIT[dataset] if split == "dev" else split


def load_split(dataset, split):
    """(queries, qrels) for 'dev' or 'test'. Queries limited to the ones judged
    in this split, which is what keeps the two splits from ever touching."""
    qrels = load_qrels(dataset, split_name(dataset, split))
    all_queries = load_queries(dataset)
    queries = {qid: all_queries[qid] for qid in qrels if qid in all_queries}
    return queries, qrels


# --------------------------------------------------------------------- run i/o

def run_path(dataset, split, system):
    return RUNS / f"{dataset}.{split}.{system}.json.gz"


def save_run(run, dataset, split, system):
    RUNS.mkdir(parents=True, exist_ok=True)
    with gzip.open(run_path(dataset, split, system), "wt") as fh:
        json.dump(run, fh)


def load_run(dataset, split, system):
    with gzip.open(run_path(dataset, split, system), "rt") as fh:
        return json.load(fh)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def ranked_ids(run_for_query):
    """A run entry is [[doc_id, score], ...] already sorted best first."""
    return [d for d, _ in run_for_query]


# --------------------------------------------------------------------- metrics
# trec_eval conventions: gain 2^rel - 1, discount log2(rank + 1), unjudged
# documents count as relevance 0, ideal ranking drawn from the judged pool.

def ndcg_at_k(ranking, rels, k=10):
    dcg = 0.0
    for i, doc in enumerate(ranking[:k]):
        rel = rels.get(doc, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 2)
    ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranking, rels, k=100):
    relevant = {d for d, r in rels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranking[:k])) / len(relevant)


def mrr_at_k(ranking, rels, k=10):
    for i, doc in enumerate(ranking[:k]):
        if rels.get(doc, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def success_at_k(ranking, rels, k=1):
    return float(any(rels.get(d, 0) > 0 for d in ranking[:k]))


METRICS = {
    "ndcg@10": lambda r, q: ndcg_at_k(r, q, 10),
    "recall@100": lambda r, q: recall_at_k(r, q, 100),
    "mrr@10": lambda r, q: mrr_at_k(r, q, 10),
    "success@1": lambda r, q: success_at_k(r, q, 1),
}


def per_query_scores(run, qrels, metric):
    """Ordered by sorted query id so paired comparisons line up across systems."""
    fn = METRICS[metric]
    qids = sorted(qrels)
    return qids, np.array([fn(ranked_ids(run.get(q, [])), qrels[q]) for q in qids])


# ---------------------------------------------------------------------- fusion

def reciprocal_rank_fusion(runs, k=60, depth=1000):
    """runs: list of {qid: [[doc, score], ...]}. Score is 1/(k + rank)."""
    fused = {}
    for qid in {q for r in runs for q in r}:
        acc = defaultdict(float)
        for run in runs:
            for rank, (doc, _) in enumerate(run.get(qid, []), start=1):
                acc[doc] += 1.0 / (k + rank)
        fused[qid] = _top(acc, depth)
    return fused


def minmax(scores):
    """Per query min max onto [0, 1]. A flat list maps to all ones, which is the
    only choice that keeps a constant scoring system from vanishing in a blend."""
    if not scores:
        return {}
    vals = [s for _, s in scores]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {d: 1.0 for d, _ in scores}
    return {d: (s - lo) / (hi - lo) for d, s in scores}


def score_blend(dense_run, sparse_run, alpha, depth=1000):
    """alpha * dense + (1 - alpha) * sparse on per query min max normalised
    scores. A document absent from one list scores 0 there."""
    fused = {}
    for qid in set(dense_run) | set(sparse_run):
        d = minmax(dense_run.get(qid, []))
        s = minmax(sparse_run.get(qid, []))
        acc = defaultdict(float)
        for doc, v in d.items():
            acc[doc] += alpha * v
        for doc, v in s.items():
            acc[doc] += (1 - alpha) * v
        fused[qid] = _top(acc, depth)
    return fused


def _top(acc, depth):
    """Sorted by score desc, doc id asc. The id tiebreak keeps runs reproducible."""
    items = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[:depth]
    return [[d, float(s)] for d, s in items]


def apply_rerank(first_stage, rerank_scores, k, depth=1000):
    """Cross encoder reorders the top k of the first stage. Ranks below k keep
    their first stage order underneath, so nDCG@10 for k >= 10 only ever sees
    reranked documents."""
    out = {}
    for qid, ranked in first_stage.items():
        head = ranked[:k]
        tail = ranked[k:depth]
        scores = rerank_scores.get(qid, {})
        head = sorted(head, key=lambda ds: (-scores.get(ds[0], -1e9), ds[0]))
        # Tail scores are pushed below every reranked document so the two
        # segments never interleave.
        floor = min([scores.get(d, 0.0) for d, _ in head], default=0.0)
        out[qid] = ([[d, float(scores.get(d, floor))] for d, _ in head]
                    + [[d, float(floor - 1000 - i)] for i, (d, _) in enumerate(tail)])
    return out


# ------------------------------------------------------------------ statistics

def paired_bootstrap(a, b, samples, seed):
    """Percentile CI on mean(a) - mean(b), resampling queries, not systems.
    Returns (diff, lo, hi, p_two_sided)."""
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(samples, n))
    diffs = (a - b)[idx].mean(axis=1)
    observed = float(a.mean() - b.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Fraction of resamples on the other side of zero, doubled. The usual
    # bootstrap two sided p, floored at 1/samples since it cannot resolve finer.
    tail = min((diffs <= 0).mean(), (diffs >= 0).mean())
    p = max(2 * float(tail), 1.0 / samples)
    return observed, float(lo), float(hi), p


def mean_ci(x, samples, seed):
    """Bootstrap CI on a single system's mean."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(samples, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


# -------------------------------------------------------------- system catalog

def system_list(tuning):
    """The systems compared for one dataset, given that dataset's dev tuning.
    Shared by the latency, evaluation and reporting steps so they cannot drift.

    Each entry is (name, first_stage, rerank_k). rerank_k None means no reranker.
    """
    best_fs = tuning["best_first_stage"]
    systems = [("bm25", "bm25", None),
               ("dense-minilm", "dense-minilm", None),
               ("dense-bge", "dense-bge", None),
               ("hybrid-rrf", "hybrid-rrf", None),
               ("hybrid-blend", "hybrid-blend", None)]
    systems += [(f"{best_fs}+ce@{k}", best_fs, k) for k in (10, 50, 100)]
    if best_fs != "bm25":
        systems.append(("bm25+ce@100", "bm25", 100))
    return systems


def best_system_on_dev(results, dataset):
    """The stack this repo recommends, picked on dev so test stays untouched."""
    dev = results[dataset]["dev"]["systems"]
    return max(dev, key=lambda s: dev[s]["ndcg@10"]["mean"])


# ------------------------------------------------------------------- embedding

def emb_path(dataset, model, kind):
    return CACHE / f"{dataset}.{model}.{kind}.npy"


def ids_path(dataset, kind):
    return CACHE / f"{dataset}.{kind}_ids.json"
