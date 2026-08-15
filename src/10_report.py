"""Render README.md from README.template.md.

The template holds the prose. Every number in it is a {{dotted.path}} into the
generated JSON, so a number that is not reproducible is a number that cannot be
written. An unresolved placeholder is a hard error, not a blank.
"""
import re

from config import DATASETS, OUTPUT, ROOT
from lib import best_system_on_dev, load_json

PLACEHOLDER = re.compile(r"\{\{([^{}]+)\}\}")


def lookup(ctx, path):
    node = ctx
    for part in path.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node[part]
    return node


def render(template, ctx):
    missing = []

    def sub(match):
        expr = match.group(1).strip()
        path, _, spec = expr.partition("|")
        try:
            value = lookup(ctx, path.strip())
        except (KeyError, IndexError, TypeError) as exc:
            missing.append(f"{expr} ({exc})")
            return match.group(0)
        return format(value, spec.strip()) if spec.strip() else str(value)

    out = PLACEHOLDER.sub(sub, template)
    if missing:
        raise SystemExit("unresolved placeholders in README.template.md:\n  "
                         + "\n  ".join(missing))
    return out


# ------------------------------------------------------------- table builders

def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def ci(entry, digits=4):
    return f"{entry['mean']:.{digits}f} [{entry['ci_lo']:.{digits}f}, {entry['ci_hi']:.{digits}f}]"


def dataset_table(audit):
    rows = []
    for a in audit:
        for split in ["dev", "test"]:
            s = a["splits"][split]
            rows.append([a["dataset"], f"{split} (`{s['official_split']}`)",
                         f"{a['corpus_docs']:,}", s["queries"],
                         s["positives_per_query_mean"],
                         f"{s['label_density_pct']:.4f}%",
                         ", ".join(f"{g}:{n:,}" for g, n in sorted(s["grades"].items()))])
    return md_table(["dataset", "split", "docs", "queries", "positives/query",
                     "label density", "grades"], rows)


def results_table(results, latency, ds):
    r = results[ds]["test"]["systems"]
    lat = latency["datasets"][ds]["systems_ms"]
    rows = []
    for name, m in r.items():
        rows.append([f"`{name}`", ci(m["ndcg@10"]), ci(m["recall@100"]),
                     ci(m["mrr@10"]), ci(m["success@1"]),
                     f"{lat[name]['p50']:.0f}", f"{lat[name]['p95']:.0f}",
                     f"${lat[name]['usd_per_1k_queries']:.3f}"])
    return md_table(["system", "nDCG@10 [95% CI]", "Recall@100 [95% CI]",
                     "MRR@10 [95% CI]", "Success@1 [95% CI]",
                     "p50 ms", "p95 ms", "$/1k queries"], rows)


def comparison_table(results, ds, pairs):
    by_key = {(c["a"], c["b"], c["metric"]): c
              for c in results[ds]["test"]["comparisons"]}
    rows = []
    for a, b in pairs:
        c = by_key.get((a, b, "ndcg@10"))
        flip = False
        if c is None:
            c = by_key[(b, a, "ndcg@10")]
            flip = True
        diff = -c["diff"] if flip else c["diff"]
        lo, hi = (-c["ci_hi"], -c["ci_lo"]) if flip else (c["ci_lo"], c["ci_hi"])
        rows.append([f"`{a}` vs `{b}`", f"{diff:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]",
                     f"{c['p']:.4f}",
                     "yes" if c["distinguishable"] else "**no**"])
    return md_table(["comparison (nDCG@10, test)", "difference", "95% CI", "p",
                     "distinguishable from noise"], rows)


def index_table(index_stats):
    order = {"bm25": 0, "minilm": 1, "bge": 2}
    rows = []
    for s in sorted(index_stats, key=lambda s: (DATASETS.index(s["dataset"]),
                                                order[s["model"]])):
        rows.append([s["dataset"], f"`{s['model']}`", f"{s['docs']:,}",
                     f"{s['index_build_s']:.1f}", f"{s['docs_per_s']:.0f}",
                     f"{s['index_mb']:.1f}", s["device"]])
    return md_table(["dataset", "index", "docs", "build s", "docs/s", "MB on disk",
                     "device"], rows)


def stage_table(latency):
    keys = sorted({k for d in latency["datasets"].values() for k in d["stages_ms"]})
    rows = []
    for k in keys:
        row = [f"`{k}`"]
        for ds in DATASETS:
            s = latency["datasets"][ds]["stages_ms"].get(k)
            row.append(f"{s['p50']:.1f} / {s['p95']:.1f}" if s else "-")
        rows.append(row)
    return md_table(["stage, p50 / p95 ms"] + DATASETS, rows)


def tuning_table(tuning):
    rows = []
    for t in tuning:
        rows.append([t["dataset"], f"`{t['best_dense']}`",
                     f"{t['alpha_selected_on_dev']:.2f}",
                     f"{t['alpha_dev_ndcg@10']:.4f}",
                     f"{t['alpha_test_ndcg@10_at_dev_choice']:.4f}",
                     f"{t['alpha_best_on_test']:.2f}",
                     f"{t['alpha_test_ndcg@10_if_tuned_on_test']:.4f}",
                     f"{t['alpha_overfit_gap']:+.4f}",
                     f"`{t['best_first_stage']}`"])
    return md_table(["dataset", "dense picked on dev", "alpha picked on dev",
                     "dev nDCG@10", "test nDCG@10 at that alpha",
                     "alpha that would have won on test", "test nDCG@10 there",
                     "gap given away by honest tuning", "first stage into the reranker"],
                    rows)


def failure_table(failures):
    modes = failures["modes"]
    rows = []
    for mode in modes:
        row = [mode]
        for ds in DATASETS:
            d = failures["datasets"][ds]
            n = d["counts"].get(mode, 0)
            row.append(f"{n} ({d['share'].get(mode, 0.0):.0f}%)" if n else "0")
        rows.append(row)
    total = ["**labelled queries**"] + [f"**{failures['datasets'][ds]['labelled']}**"
                                        for ds in DATASETS]
    return md_table(["failure mode"] + DATASETS, rows + [total])


def examples_block(failures, ds, per_mode=1):
    d = failures["datasets"][ds]
    out = []
    for mode in failures["modes"]:
        for ex in d["examples"].get(mode, [])[:per_mode]:
            out.append(
                f"- **{mode}** — `{ex['qid']}` \"{ex['query'].strip()}\"  \n"
                f"  nDCG@10: best stack {ex['ndcg_best']:.3f}, "
                f"BM25 {ex['ndcg_bm25']:.3f}, dense {ex['ndcg_dense']:.3f}  \n"
                f"  gold: \"{ex['gold_snippet'].strip()[:180]}...\"  \n"
                f"  top hit: \"{ex['top1_snippet'].strip()[:180]}...\"")
    return "\n".join(out)


# ------------------------------------------------------------------- context

def build_context():
    audit = load_json(OUTPUT / "audit.json")
    results = load_json(OUTPUT / "results.json")
    latency = load_json(OUTPUT / "latency.json")
    tuning = load_json(OUTPUT / "tuning.json")
    index_stats = load_json(OUTPUT / "index_stats.json")
    failures = load_json(OUTPUT / "failure_modes.json")

    calc = {}
    for ds in DATASETS:
        sysres = results[ds]["test"]["systems"]
        lat = latency["datasets"][ds]["systems_ms"]
        best = best_system_on_dev(results, ds)
        cmp_by = {(c["a"], c["b"]): c for c in results[ds]["test"]["comparisons"]
                  if c["metric"] == "ndcg@10"}

        def diff(a, b):
            c = cmp_by.get((a, b))
            if c:
                return c
            c = dict(cmp_by[(b, a)])
            c["diff"], c["ci_lo"], c["ci_hi"] = -c["diff"], -c["ci_hi"], -c["ci_lo"]
            return c

        best_fs = [f for n, f, k in
                   __import__("lib").system_list(next(t for t in tuning
                                                      if t["dataset"] == ds))
                   if n == best][0]
        calc[ds] = {
            "best_system": best,
            "best_first_stage": best_fs,
            "best_ndcg": sysres[best]["ndcg@10"]["mean"],
            "best_ndcg_lo": sysres[best]["ndcg@10"]["ci_lo"],
            "best_ndcg_hi": sysres[best]["ndcg@10"]["ci_hi"],
            "bm25_ndcg": sysres["bm25"]["ndcg@10"]["mean"],
            "best_vs_bm25": diff(best, "bm25"),
            "best_vs_first_stage": diff(best, best_fs),
            "hybrid_vs_bm25": diff("hybrid-blend", "bm25"),
            "hybrid_vs_dense": diff("hybrid-blend", f"dense-{[t for t in tuning if t['dataset'] == ds][0]['best_dense']}"),
            "ce100_vs_ce10": diff(f"{best_fs}+ce@100", f"{best_fs}+ce@10")
                             if f"{best_fs}+ce@100" in sysres else None,
            "best_p95": lat[best]["p95"],
            "bm25_p95": lat["bm25"]["p95"],
            "p95_ratio": round(lat[best]["p95"] / lat["bm25"]["p95"], 1),
            "best_usd": lat[best]["usd_per_1k_queries"],
            "bm25_usd": lat["bm25"]["usd_per_1k_queries"],
            "usd_ratio": round(lat[best]["usd_per_1k_queries"]
                               / lat["bm25"]["usd_per_1k_queries"], 1),
            "recall_ceiling": results[ds]["test"]["recall_ceiling_at_k100"],
            "first_stage_recall100": sysres[best_fs]["recall@100"]["mean"],
            "bm25_recall100": sysres["bm25"]["recall@100"]["mean"],
            "queries": results[ds]["test"]["queries"],
            "zero_ndcg": failures["datasets"][ds]["total_zero_ndcg"],
            "zero_ndcg_pct": round(100 * failures["datasets"][ds]["total_zero_ndcg"]
                                   / results[ds]["test"]["queries"], 1),
            "labelled": failures["datasets"][ds]["labelled"],
        }

    return {
        "audit": {a["dataset"]: a for a in audit},
        "results": results,
        "latency": latency,
        "tuning": {t["dataset"]: t for t in tuning},
        "failures": failures,
        "calc": calc,
        "machine": latency["machine"],
        "cost": latency["cost"],
        "timed_queries": latency["queries_timed"],
        "table": {
            "datasets": dataset_table(audit),
            "index": index_table(index_stats),
            "stages": stage_table(latency),
            "tuning": tuning_table(tuning),
            "failures": failure_table(failures),
            **{f"results_{ds}": results_table(results, latency, ds) for ds in DATASETS},
            **{f"examples_{ds}": examples_block(failures, ds) for ds in DATASETS},
            **{f"comparisons_{ds}": comparison_table(
                results, ds,
                [(calc[ds]["best_system"], "bm25"),
                 (calc[ds]["best_system"], calc[ds]["best_first_stage"]),
                 ("hybrid-blend", "bm25"),
                 ("hybrid-blend", "hybrid-rrf"),
                 ("dense-bge", "dense-minilm"),
                 ("dense-bge", "bm25"),
                 (f"{calc[ds]['best_first_stage']}+ce@100",
                  f"{calc[ds]['best_first_stage']}+ce@10"),
                 ("bm25+ce@100", "bm25")]) for ds in DATASETS},
        },
    }


if __name__ == "__main__":
    template = (ROOT / "README.template.md").read_text()
    (ROOT / "README.md").write_text(render(template, build_context()))
    print(f"wrote {ROOT / 'README.md'}")
