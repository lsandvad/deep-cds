"""Step 4: detection rate per tool and UHGG mapping category (Wilson 95% CI), UpSet plots, summary figure.

Detection = at least one CDS >= 61 bp predicted on the read (either strand); population = reads >= 61 bp.
Inputs:  WORK/detection/detection.tsv.gz, WORK/mapping/read_categories.tsv.gz (same read order, checked)
Outputs: RES/detection_rates.tsv, RES/upset_<category>.tsv,
         FIG/detection_rates.{pdf,png}, FIG/upset_<category>.{pdf,png}
Usage: python 08_analysis.py [--work /tmp/nrt204/real_metagenome_poc_work]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = "/tmp/nrt204/FragmentPredictor/real_metagenome_poc/results", "/tmp/nrt204/FragmentPredictor/real_metagenome_poc/figures"
NAMES = {"deepcds_S": "DeepCDS S", "deepcds_SI": "DeepCDS S+I", "prodigal": "MetaProdigal",
         "fgs_complete": "FGS (Complete)", "fgs_illumina_5": "FGS (Illumina 5)", "fgs_illumina_10": "FGS (Illumina 10)"}
CATEGORIES = ["mappable", "intermediate", "unmappable"]
UPSET_TOOLS = ["deepcds_S", "deepcds_SI", "prodigal", "fgs_complete"]


def draw_upset(ind, title, path_stem):
    """UpSet-style plot in plain matplotlib (upsetplot 0.9 does not work with pandas 3): intersection sizes as bars
    above a dot matrix, sorted by size; percentages relative to all reads in the group."""
    names = list(ind.columns)
    combos = ind.value_counts().sort_values(ascending=False)
    n = len(ind)
    fig, (ax_bar, ax_dot) = plt.subplots(2, 1, figsize=(max(6, 0.55 * len(combos) + 2), 5.5), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.05})
    x = np.arange(len(combos))
    ax_bar.bar(x, combos.values, color="0.25", width=0.6)
    for xi, v in zip(x, combos.values):
        ax_bar.text(xi, v, f"{v / n:.1%}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax_bar.set_ylabel("Reads")
    ax_bar.set_title(title)
    ax_bar.spines[["top", "right"]].set_visible(False)
    for yi in range(len(names)):
        ax_dot.scatter(x, [yi] * len(x), color="0.85", s=40, zorder=1)
    for xi, combo in zip(x, combos.index):
        on = [yi for yi, flag in enumerate(combo) if flag]
        if on:
            ax_dot.scatter([xi] * len(on), on, color="0.15", s=40, zorder=2)
            ax_dot.plot([xi, xi], [min(on), max(on)], color="0.15", linewidth=1.5, zorder=2)
    ax_dot.set_yticks(range(len(names)), names)
    ax_dot.set_ylim(-0.6, len(names) - 0.4)
    ax_dot.set_xticks([])
    ax_dot.spines[["top", "right", "bottom", "left"]].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/real_metagenome_poc_work")
    args = parser.parse_args()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    det = pd.read_csv(f"{args.work}/detection/detection.tsv.gz", sep="\t")
    cat = pd.read_csv(f"{args.work}/mapping/read_categories.tsv.gz", sep="\t", usecols=["read_id", "category"])
    assert len(det) == len(cat) and (det["read_id"].values == cat["read_id"].values).all(), "read order differs"
    det["category"] = cat["category"].values
    tools = [t for t in NAMES if t in det.columns]

    rows = []
    for group in ["all"] + CATEGORIES:
        sub = det if group == "all" else det[det["category"] == group]
        n = len(sub)
        for t in tools:
            k = int(sub[t].sum())
            lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
            rows.append({"category": group, "n_reads": n, "tool": NAMES[t], "detected": k,
                         "rate": k / n, "ci_low": lo, "ci_high": hi})
    rates = pd.DataFrame(rows)
    rates.to_csv(f"{RES}/detection_rates.tsv", sep="\t", index=False)
    print(rates.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # summary figure: detection rate by category, tools side by side, Wilson CI
    fig, ax = plt.subplots(figsize=(8, 4))
    groups = ["all"] + CATEGORIES
    width = 0.8 / len(tools)
    for i, t in enumerate(tools):
        r = rates[rates["tool"] == NAMES[t]].set_index("category").loc[groups]
        x = np.arange(len(groups)) + (i - (len(tools) - 1) / 2) * width
        ax.bar(x, r["rate"], width, label=NAMES[t],
               yerr=[r["rate"] - r["ci_low"], r["ci_high"] - r["rate"]], capsize=2)
    n = rates.drop_duplicates("category").set_index("category")["n_reads"]
    ax.set_xticks(np.arange(len(groups)), [f"{g}\n(n={n[g]:,})" for g in groups])
    ax.set_ylabel("Reads with a predicted CDS (>= 61 bp)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}/detection_rates.{ext}", dpi=300)
    plt.close(fig)

    # UpSet: which tools detect which reads, per category (DeepCDS S, DeepCDS S+I, MetaProdigal, FGS Complete)
    up_tools = [t for t in UPSET_TOOLS if t in tools]
    for group in ["all"] + CATEGORIES:
        sub = det if group == "all" else det[det["category"] == group]
        ind = sub[up_tools].astype(bool).rename(columns=NAMES)
        combos = ind.value_counts().rename("reads").reset_index()
        combos["fraction"] = combos["reads"] / len(sub)
        combos.to_csv(f"{RES}/upset_{group}.tsv", sep="\t", index=False)
        draw_upset(ind, f"{group} reads (n={len(sub):,})", f"{FIG}/upset_{group}")


if __name__ == "__main__":
    main()
