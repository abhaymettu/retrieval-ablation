"""The hero figure (quality against p95 latency, marker area is dollars) and the
blend weight sensitivity figure.

Colour encodes the first stage, marker shape encodes whether a cross encoder is
attached, and every point is directly labelled, so nothing here is carried by
colour alone.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DATASETS, OUTPUT
from lib import load_json

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dedcd5"
# Slots 1 to 3 of the reference categorical palette. Validated for all pairs in
# light mode, which is what a scatter needs.
FAMILY = {"sparse": "#2a78d6", "dense": "#eb6834", "hybrid": "#1baf7a"}


def family(system):
    stage = system.split("+")[0]
    if stage == "bm25":
        return "sparse"
    return "dense" if stage.startswith("dense-") else "hybrid"


def pareto_front(points):
    """Systems where nothing is both faster and better."""
    front = []
    for name, x, y in sorted(points, key=lambda p: p[1]):
        if not front or y > front[-1][2]:
            front.append((name, x, y))
    return front


def hero(results, latency):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.0 * len(DATASETS), 5.0),
                             facecolor=SURFACE)
    costs = [v["usd_per_1k_queries"] for ds in DATASETS
             for v in latency["datasets"][ds]["systems_ms"].values()]
    cmin, cmax = min(costs), max(costs)

    def area(c):
        return 60 + 900 * (c - cmin) / (cmax - cmin)

    for ax, ds in zip(np.atleast_1d(axes), DATASETS):
        ax.set_facecolor(SURFACE)
        systems = results[ds]["test"]["systems"]
        lat = latency["datasets"][ds]["systems_ms"]
        points = []
        for name, m in systems.items():
            x, y = lat[name]["p95"], m["ndcg@10"]["mean"]
            points.append((name, x, y))
            ax.scatter(x, y, s=area(lat[name]["usd_per_1k_queries"]),
                       color=FAMILY[family(name)], alpha=0.85,
                       marker="^" if "+ce@" in name else "o",
                       edgecolors=SURFACE, linewidths=2, zorder=3)
            lo, hi = m["ndcg@10"]["ci_lo"], m["ndcg@10"]["ci_hi"]
            ax.plot([x, x], [lo, hi], color=FAMILY[family(name)], alpha=0.45,
                    linewidth=2, zorder=2)
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 13),
                        ha="center", fontsize=7.5, color=INK_2, zorder=4)

        front = pareto_front(points)
        ax.plot([p[1] for p in front], [p[2] for p in front], color=GRID,
                linewidth=2, linestyle="--", zorder=1)

        ax.set_xscale("log")
        ax.set_xlabel("p95 latency per query (ms, log scale)", fontsize=9, color=INK_2)
        ax.set_title(f"{ds}  ({results[ds]['test']['queries']} test queries)",
                     fontsize=11, color=INK, loc="left")
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8)

    np.atleast_1d(axes)[0].set_ylabel("nDCG@10 on test (bars are 95% bootstrap CI)",
                                      fontsize=9, color=INK_2)

    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=9,
                          label=n) for n, c in FAMILY.items()]
    handles += [plt.Line2D([], [], marker="^", linestyle="", color=INK_2,
                           markersize=9, label="cross encoder attached"),
                plt.Line2D([], [], marker="o", linestyle="", color="none",
                           markerfacecolor="none", markeredgecolor=INK_2,
                           markersize=11,
                           label=f"marker area = ${cmin:.3f} to ${cmax:.3f} per 1k queries")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8.5, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("What each retrieval layer buys, and what it costs",
                 fontsize=13, color=INK, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    path = OUTPUT / "pareto.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {path}")


def sensitivity(tuning):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.4 * len(DATASETS), 3.8),
                             facecolor=SURFACE, sharey=False)
    for ax, row in zip(np.atleast_1d(axes), tuning):
        ax.set_facecolor(SURFACE)
        a = [r["alpha"] for r in row["alpha_sweep"]]
        ax.plot(a, [r["dev_ndcg@10"] for r in row["alpha_sweep"]],
                color="#2a78d6", linewidth=2, label="dev (used to choose)")
        ax.plot(a, [r["test_ndcg@10"] for r in row["alpha_sweep"]],
                color="#eb6834", linewidth=2, linestyle=(0, (5, 2)),
                label="test (never used to choose)")
        pick = row["alpha_selected_on_dev"]
        ax.axvline(pick, color=INK_2, linewidth=1, alpha=0.6)
        ax.annotate(f"alpha {pick:.2f} chosen on dev", (pick, ax.get_ylim()[0]),
                    textcoords="offset points", xytext=(4, 6), fontsize=7.5,
                    color=INK_2)
        ax.set_title(row["dataset"], fontsize=11, color=INK, loc="left")
        ax.set_xlabel("alpha (0 = pure BM25, 1 = pure dense)", fontsize=9, color=INK_2)
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8)
    np.atleast_1d(axes)[0].set_ylabel("nDCG@10", fontsize=9, color=INK_2)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.suptitle("Blend weight sensitivity: how much a test tuned hybrid would have flattered itself",
                 fontsize=12, color=INK, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUTPUT / "weight_sensitivity.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    hero(load_json(OUTPUT / "results.json"), load_json(OUTPUT / "latency.json"))
    sensitivity(load_json(OUTPUT / "tuning.json"))
