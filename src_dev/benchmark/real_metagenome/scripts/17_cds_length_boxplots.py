"""Step 10: boxplots of the predicted CDS lengths per tool.

From the aa-length histogram of 15_length_distributions.py (no re-read of the CDS tables). Boxes: quartiles; whiskers:
the most extreme lengths within 1.5 x IQR of the box; points: each length outside the whiskers that occurs at all.
Outputs: FIG/cds_length_boxplots.{pdf,png}, RES/cds_length_boxplot_stats.tsv
Usage: python 17_cds_length_boxplots.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = "/tmp/nrt204/FragmentPredictor/real_metagenome_poc/results", "/tmp/nrt204/FragmentPredictor/real_metagenome_poc/figures"
sys.path.insert(0, f"{HERE}/../../eval")
from plot_config import MODEL_COLORS, MODEL_DISPLAY_NAMES, FONT_SIZES  # noqa: E402

TOOLS = {"DeepCDS N": "deep_cds_no_errors", "DeepCDS S": "deep_cds_substitution", "DeepCDS S+I": "deep_cds_errors",
         "MetaProdigal": "prodigal", "FGS (Complete)": "fgs_complete", "FGS (Illumina 5)": "fgs_illumina_5",
         "FGS (Illumina 10)": "fgs_illumina_10"}


def box_stats(h):
    """Tukey box statistics of a length histogram (index = length, values = counts)."""
    h = h[h > 0]
    c = h.cumsum() / h.sum()
    q = {p: h.index[np.searchsorted(c.to_numpy(), p)] for p in (0.25, 0.5, 0.75)}
    iqr = q[0.75] - q[0.25]
    inside = h.index[(h.index >= q[0.25] - 1.5 * iqr) & (h.index <= q[0.75] + 1.5 * iqr)]
    lo, hi = inside.min(), inside.max()
    return {"q1": q[0.25], "med": q[0.5], "q3": q[0.75], "whislo": lo, "whishi": hi,
            "mean": (h.index * h).sum() / h.sum(), "fliers": h.index[(h.index < lo) | (h.index > hi)].to_numpy(),
            "below_q1": h[h.index < q[0.25]].sum() / h.sum(), "n": int(h.sum())}


def main():
    cds = pd.read_csv(f"{RES}/length_distribution_cds.tsv", sep="\t", index_col="length_aa")
    stats = [box_stats(cds[t]) for t in TOOLS]
    pd.DataFrame([{k: v for k, v in s.items() if k != "fliers"} | {"tool": t, "fliers": ",".join(map(str, s["fliers"]))}
                  for t, s in zip(TOOLS, stats)]).set_index("tool").to_csv(f"{RES}/cds_length_boxplot_stats.tsv", sep="\t")
    for t, s in zip(TOOLS, stats):
        print(f"{t}: Q1 {s['q1']}, median {s['med']}, Q3 {s['q3']}, whiskers {s['whislo']}-{s['whishi']}, "
              f"mean {s['mean']:.2f}, < Q1 {s['below_q1']:.3f}")

    plt.rcParams.update(FONT_SIZES)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(TOOLS))[::-1]
    b = ax.bxp(stats, positions=y, vert=False, widths=0.6, patch_artist=True, showmeans=False,
               medianprops={"color": "black", "linewidth": 2}, whiskerprops={"linewidth": 1.2},
               capprops={"linewidth": 1.2}, flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "#555555",
                                                        "markeredgecolor": "none"})
    for patch, k in zip(b["boxes"], TOOLS.values()):
        patch.set_facecolor(MODEL_COLORS[k])
        patch.set_edgecolor("black")
    ax.set_yticks(y, [MODEL_DISPLAY_NAMES[k] for k in TOOLS.values()])
    ax.set_xlabel("Predicted CDS length (aa)")
    ax.set_xticks(np.arange(18, 31, 2))
    ax.set_xlim(17.5, 30.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}/cds_length_boxplots.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")


if __name__ == "__main__":
    main()
