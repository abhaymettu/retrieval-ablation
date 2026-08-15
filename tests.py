"""Plain assert tests. Run with: make test

Covers the four things that would silently corrupt every number in the README:
the qrels join, the metric implementations, the fusion functions, and the
guarantee that dev and test never mix.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

from config import DATASETS, DEV_SPLIT, OUTPUT
from lib import (apply_rerank, load_corpus, load_json, load_qrels, load_run,
                 load_split, minmax, mrr_at_k, ndcg_at_k, paired_bootstrap,
                 ranked_ids, recall_at_k, reciprocal_rank_fusion, run_path,
                 score_blend, split_name, success_at_k)

PASSED = []


def check(name):
    def deco(fn):
        PASSED.append((name, fn))
        return fn
    return deco


# ------------------------------------------------------------------ metrics
# Fixture: rels {a: 2, b: 1, c: 0}, ranking [x, b, a].
# DCG  = 0 + (2^1 - 1)/log2(3) + (2^2 - 1)/log2(4) = 0.63092975 + 1.5
# IDCG = (2^2 - 1)/log2(2) + (2^1 - 1)/log2(3)     = 3 + 0.63092975
# nDCG = 2.13092975 / 3.63092975 = 0.58688...

@check("ndcg matches a hand computed graded example")
def _():
    rels = {"a": 2, "b": 1, "c": 0}
    got = ndcg_at_k(["x", "b", "a"], rels, k=10)
    assert abs(got - 0.5868818) < 1e-6, got


@check("ndcg is 1.0 for the ideal ranking and 0.0 with nothing relevant")
def _():
    rels = {"a": 2, "b": 1}
    assert abs(ndcg_at_k(["a", "b"], rels, 10) - 1.0) < 1e-12
    assert ndcg_at_k(["z", "y"], rels, 10) == 0.0
    assert ndcg_at_k(["a"], {"c": 0}, 10) == 0.0     # no positives, no credit


@check("ndcg respects the cutoff")
def _():
    rels = {"a": 1}
    assert ndcg_at_k(["x"] * 9 + ["a"], rels, 10) > 0
    assert ndcg_at_k(["x"] * 10 + ["a"], rels, 10) == 0.0


@check("recall, mrr and success match hand computed values")
def _():
    rels = {"a": 1, "b": 1, "c": 2, "d": 0}
    ranking = ["x", "a", "y", "c"]
    assert recall_at_k(ranking, rels, 100) == 2 / 3        # a and c of {a, b, c}
    assert recall_at_k(ranking, rels, 2) == 1 / 3
    assert mrr_at_k(ranking, rels, 10) == 0.5              # first hit at rank 2
    assert mrr_at_k(["z"], rels, 10) == 0.0
    assert success_at_k(ranking, rels, 1) == 0.0
    assert success_at_k(["c"], rels, 1) == 1.0
    assert recall_at_k(["a"], {"d": 0}, 100) == 0.0        # no positives


# ------------------------------------------------------------------- fusion

@check("rrf matches hand computed reciprocal ranks")
def _():
    r1 = {"q": [["a", 9.0], ["b", 8.0]]}
    r2 = {"q": [["b", 0.9], ["c", 0.8]]}
    fused = dict(reciprocal_rank_fusion([r1, r2], k=60)["q"])
    assert abs(fused["a"] - 1 / 61) < 1e-12
    assert abs(fused["b"] - (1 / 62 + 1 / 61)) < 1e-12
    assert abs(fused["c"] - 1 / 62) < 1e-12
    assert ranked_ids(reciprocal_rank_fusion([r1, r2], k=60)["q"])[0] == "b"


@check("minmax normalises to [0, 1] and survives a flat list")
def _():
    m = minmax([["a", 5.0], ["b", 1.0], ["c", 3.0]])
    assert m == {"a": 1.0, "b": 0.0, "c": 0.5}
    assert minmax([["a", 2.0], ["b", 2.0]]) == {"a": 1.0, "b": 1.0}
    assert minmax([]) == {}


@check("score blend is a weighted sum of normalised scores")
def _():
    dense = {"q": [["a", 1.0], ["b", 0.0]]}
    sparse = {"q": [["b", 10.0], ["c", 0.0]]}
    got = dict(score_blend(dense, sparse, alpha=0.7)["q"])
    assert abs(got["a"] - 0.7) < 1e-12          # 0.7 * 1 + 0.3 * absent
    assert abs(got["b"] - 0.3) < 1e-12          # 0.7 * 0 + 0.3 * 1
    assert abs(got["c"] - 0.0) < 1e-12
    # alpha 1.0 and 0.0 must reproduce the input orderings exactly
    assert ranked_ids(score_blend(dense, sparse, 1.0)["q"])[0] == "a"
    assert ranked_ids(score_blend(dense, sparse, 0.0)["q"])[0] == "b"


@check("blend and rrf are deterministic under ties")
def _():
    r = {"q": [["b", 1.0], ["a", 1.0]]}
    assert ranked_ids(reciprocal_rank_fusion([r, r], 60)["q"]) == ["b", "a"]
    twice = [ranked_ids(score_blend(r, r, 0.5)["q"]) for _ in range(2)]
    assert twice[0] == twice[1]


# ------------------------------------------------------------------ reranking

@check("rerank reorders only the top k and keeps the tail below it")
def _():
    first = {"q": [[c, 10.0 - i] for i, c in enumerate("abcde")]}
    scores = {"q": {"a": 0.0, "b": 5.0, "c": 1.0}}
    out = ranked_ids(apply_rerank(first, scores, k=3)["q"])
    assert out == ["b", "c", "a", "d", "e"], out
    # k = 0 leaves the run untouched in order
    assert ranked_ids(apply_rerank(first, scores, k=0)["q"]) == list("abcde")


@check("rerank cannot invent documents the first stage never returned")
def _():
    first = {"q": [["a", 1.0]]}
    out = ranked_ids(apply_rerank(first, {"q": {"a": 1.0, "zzz": 99.0}}, k=10)["q"])
    assert out == ["a"], out


# ---------------------------------------------------------------- statistics

@check("paired bootstrap brackets a real difference and not a null one")
def _():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 500)
    diff, lo, hi, p = paired_bootstrap(a + 0.2, a, 2000, 1)
    assert lo > 0 and hi > 0 and p < 0.01, (diff, lo, hi, p)
    diff, lo, hi, p = paired_bootstrap(a, a, 2000, 1)
    assert diff == 0.0 and lo == 0.0 and hi == 0.0


# ------------------------------------------------------------- qrels join

@check("qrels join: every judged query has text, every judged doc has a body")
def _():
    for ds in DATASETS:
        corpus = load_corpus(ds)
        for split in ["dev", "test"]:
            queries, qrels = load_split(ds, split)
            assert set(queries) == set(qrels), f"{ds}/{split}: query text missing"
            assert all(queries[q] for q in queries), f"{ds}/{split}: empty query"
            missing = [d for v in qrels.values() for d in v if d not in corpus]
            assert not missing, f"{ds}/{split}: {len(missing)} qrel docs not in corpus"


@check("qrels join: relevance grades are non negative integers")
def _():
    for ds in DATASETS:
        for split in ["dev", "test"]:
            _, qrels = load_split(ds, split)
            grades = {r for v in qrels.values() for r in v.values()}
            assert grades and all(isinstance(g, int) and g >= 0 for g in grades), grades


# ------------------------------------------------------- split hygiene guard

@check("dev and test query ids never overlap")
def _():
    for ds in DATASETS:
        dev = set(load_qrels(ds, split_name(ds, "dev")))
        test = set(load_qrels(ds, split_name(ds, "test")))
        assert not dev & test, f"{ds}: {len(dev & test)} shared query ids"
        assert split_name(ds, "test") == "test"
        assert DEV_SPLIT[ds] != "test"


@check("every tuned hyperparameter is the dev argmax, not the test argmax")
def _():
    path = OUTPUT / "tuning.json"
    if not path.exists():
        return print("      skipped, run make all first")
    for row in load_json(path):
        sweep = row["alpha_sweep"]
        dev_best = max(sweep, key=lambda r: r["dev_ndcg@10"])
        assert row["alpha_selected_on_dev"] == dev_best["alpha"], row["dataset"]
        assert row["best_dense"] == max(row["dense_dev_ndcg@10"],
                                        key=row["dense_dev_ndcg@10"].get)
        assert row["best_first_stage"] == max(row["first_stage_dev_ndcg@10"],
                                              key=row["first_stage_dev_ndcg@10"].get)


@check("runs only contain documents from that dataset's corpus")
def _():
    for ds in DATASETS:
        if not run_path(ds, "test", "bm25").exists():
            return print("      skipped, run make all first")
        corpus = set(load_corpus(ds))
        for system in ["bm25", "dense-bge", "hybrid-blend"]:
            run = load_run(ds, "test", system)
            sample = [d for q in list(run)[:20] for d, _ in run[q][:50]]
            assert set(sample) <= corpus, f"{ds}/{system}: foreign doc ids"


if __name__ == "__main__":
    failures = 0
    for name, fn in PASSED:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(PASSED) - failures}/{len(PASSED)} passed")
    sys.exit(1 if failures else 0)
