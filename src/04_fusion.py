"""Hybrid runs: reciprocal rank fusion, and a normalised score blend whose
weight is swept on dev and frozen before test is touched.

Every selection made here (which dense model, which alpha, which first stage
feeds the reranker) is made on dev, per dataset, and written to
output/tuning.json so the README can say where each number came from.
"""
import numpy as np

from config import (BLEND_ALPHAS, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, DATASETS,
                    EMBED_MODELS, OUTPUT, RETRIEVAL_DEPTH, RRF_K)
from lib import (load_run, load_split, per_query_scores, reciprocal_rank_fusion,
                 save_json, save_run, score_blend)


def ndcg(run, qrels):
    return float(per_query_scores(run, qrels, "ndcg@10")[1].mean())


def tune(dataset):
    _, dev_qrels = load_split(dataset, "dev")
    dev = {s: load_run(dataset, "dev", s) for s in
           ["bm25"] + [f"dense-{m}" for m in EMBED_MODELS]}

    dense_scores = {m: ndcg(dev[f"dense-{m}"], dev_qrels) for m in EMBED_MODELS}
    best_dense = max(dense_scores, key=dense_scores.get)

    # Alpha sweep on dev only. The test curve is computed too, but purely so the
    # README can show what tuning on test would have bought, never to pick.
    _, test_qrels = load_split(dataset, "test")
    test_bm25 = load_run(dataset, "test", "bm25")
    test_dense = load_run(dataset, "test", f"dense-{best_dense}")

    sweep = []
    for alpha in BLEND_ALPHAS:
        dev_run = score_blend(dev[f"dense-{best_dense}"], dev["bm25"], alpha, RETRIEVAL_DEPTH)
        test_run = score_blend(test_dense, test_bm25, alpha, RETRIEVAL_DEPTH)
        sweep.append({"alpha": alpha,
                      "dev_ndcg@10": round(ndcg(dev_run, dev_qrels), 4),
                      "test_ndcg@10": round(ndcg(test_run, test_qrels), 4)})
        print(f"  alpha {alpha:.2f}  dev {sweep[-1]['dev_ndcg@10']:.4f}  "
              f"test {sweep[-1]['test_ndcg@10']:.4f}")

    best = max(sweep, key=lambda r: r["dev_ndcg@10"])
    alpha = best["alpha"]

    for split, dr, sr in [("dev", dev[f"dense-{best_dense}"], dev["bm25"]),
                          ("test", test_dense, test_bm25)]:
        save_run(score_blend(dr, sr, alpha, RETRIEVAL_DEPTH), dataset, split, "hybrid-blend")
        save_run(reciprocal_rank_fusion([dr, sr], RRF_K, RETRIEVAL_DEPTH),
                 dataset, split, "hybrid-rrf")

    # Which first stage feeds the cross encoder, again decided on dev.
    dev_all = dict(dev)
    dev_all["hybrid-blend"] = load_run(dataset, "dev", "hybrid-blend")
    dev_all["hybrid-rrf"] = load_run(dataset, "dev", "hybrid-rrf")
    dev_ndcg = {s: round(ndcg(r, dev_qrels), 4) for s, r in dev_all.items()}
    best_first_stage = max(dev_ndcg, key=dev_ndcg.get)

    peak_test = max(sweep, key=lambda r: r["test_ndcg@10"])
    return {
        "dataset": dataset,
        "dev_split_used": dataset,
        "dense_dev_ndcg@10": {m: round(v, 4) for m, v in dense_scores.items()},
        "best_dense": best_dense,
        "alpha_sweep": sweep,
        "alpha_selected_on_dev": alpha,
        "alpha_dev_ndcg@10": best["dev_ndcg@10"],
        "alpha_test_ndcg@10_at_dev_choice": best["test_ndcg@10"],
        "alpha_best_on_test": peak_test["alpha"],
        "alpha_test_ndcg@10_if_tuned_on_test": peak_test["test_ndcg@10"],
        "alpha_overfit_gap": round(peak_test["test_ndcg@10"] - best["test_ndcg@10"], 4),
        "alpha_dev_spread": round(max(r["dev_ndcg@10"] for r in sweep)
                                  - min(r["dev_ndcg@10"] for r in sweep), 4),
        "first_stage_dev_ndcg@10": dev_ndcg,
        "best_first_stage": best_first_stage,
    }


if __name__ == "__main__":
    out = []
    for ds in DATASETS:
        print(f"\n{ds}: sweeping blend weight on dev")
        r = tune(ds)
        out.append(r)
        print(f"  best dense {r['best_dense']}, alpha {r['alpha_selected_on_dev']}, "
              f"first stage {r['best_first_stage']}")
    save_json(out, OUTPUT / "tuning.json")
