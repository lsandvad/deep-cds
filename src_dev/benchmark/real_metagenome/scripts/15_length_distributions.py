"""Step 9: length distributions of the reads >= 61 bp and of the predicted CDS >= 61 bp per tool.

Read lengths: the 128 read chunks (WORK/chunks, all reads >= 61 bp). CDS lengths: the per-chunk CDS tables of
12_diamond_full_sample.py, for the three FGS models those of 18_fgs_rf0_last_codon.py (last codon restored);
aa_len = translated length without a terminal stop codon.
Outputs: RES/length_distribution_{reads,cds}.tsv, FIG/length_distributions.{pdf,png}
Usage: python 15_length_distributions.py [--procs 64]
"""
import argparse
import os
import sys
from multiprocessing import Pool

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
N_CHUNKS = 128
MAX_LEN = 200


def read_lengths(path):
    h = np.zeros(MAX_LEN + 1, dtype=np.int64)
    with open(path) as fh:
        for line in fh:
            if line[0] != ">":
                h[min(len(line.rstrip()), MAX_LEN)] += 1
    return h


def cds_lengths(paths):
    """aa-length histogram per tool; FGS from the step-18 table (paths[1]), all others from step 12."""
    df = pd.read_csv(paths[0], sep="\t", usecols=["tool", "aa_len"])
    df = pd.concat([df[~df["tool"].str.startswith("FGS")], pd.read_csv(paths[1], sep="\t", usecols=["tool", "aa_len"])])
    return {t: np.bincount(np.minimum(g["aa_len"].to_numpy(), MAX_LEN), minlength=MAX_LEN + 1)
            for t, g in df.groupby("tool")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/real_metagenome_poc_work")
    parser.add_argument("--procs", type=int, default=64)
    args = parser.parse_args()

    with Pool(args.procs) as pool:
        reads = sum(pool.map(read_lengths, [f"{args.work}/chunks/chunk_{i:03d}.fa" for i in range(N_CHUNKS)]))
        res = pool.map(cds_lengths, [(f"{args.work}/diamond_full/chunks/cds_{i:03d}.tsv.gz",
                                     f"{args.work}/diamond_fgs_rf0/chunks/cds_{i:03d}.tsv.gz") for i in range(N_CHUNKS)])
    cds = pd.DataFrame({t: sum(r[t] for r in res) for t in TOOLS})
    reads = pd.Series(reads, name="reads").rename_axis("length_bp")
    reads[reads > 0].to_csv(f"{RES}/length_distribution_reads.tsv", sep="\t")
    cds.index.name = "length_aa"
    cds[cds.sum(axis=1) > 0].to_csv(f"{RES}/length_distribution_cds.tsv", sep="\t")

    n = reads.sum()
    print(f"reads: {n:,}; 90 bp: {reads[90] / n:.4f}; < 90 bp: {reads[:90].sum() / n:.4f}")
    for t in TOOLS:
        h = cds[t]
        c = np.cumsum(h)
        print(f"{t}: {c.iloc[-1]:,} CDS, median {np.searchsorted(c, (c.iloc[-1] + 1) / 2)} aa, "
              f"20-29 aa {h[20:30].sum() / c.iloc[-1]:.3f}, 30 aa {h[30] / c.iloc[-1]:.3f}, > 30 aa {h[31:].sum() / c.iloc[-1]:.4f}")

    plt.rcParams.update(FONT_SIZES)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1.35]})
    x = np.arange(61, 91)
    ax1.bar(x, reads[x] / n * 100, width=0.8, color="#555555")
    ax1.set_yscale("log")
    ax1.set_xlabel("Read length (bp)")
    ax1.set_ylabel("Reads (%)")
    ax1.set_title("A  Read lengths", loc="left")
    ax1.set_xticks([61, 70, 80, 90])
    ax1.annotate(f"{reads[90] / n * 100:.1f}%", (90, reads[90] / n * 100), xytext=(-8, 0), textcoords="offset points",
                 ha="right", va="center", fontsize=FONT_SIZES["legend.fontsize"])

    bins = {"<= 27 aa": slice(0, 28), "28 aa": slice(28, 29), "29 aa": slice(29, 30), "30 aa": slice(30, 31)}
    shades = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    names = [MODEL_DISPLAY_NAMES[k] for k in TOOLS.values()]
    y = np.arange(len(TOOLS))[::-1]
    left = np.zeros(len(TOOLS))
    for (label, sl), col in zip(bins.items(), shades):
        frac = np.array([cds[t].iloc[sl].sum() / cds[t].sum() * 100 for t in TOOLS])
        ax2.barh(y, frac, left=left, height=0.7, color=col, edgecolor="white", linewidth=1, label=label.replace("<=", "≤"))
        for yi, l, f in zip(y, left, frac):
            if f >= 8:
                ax2.text(l + f / 2, yi, f"{f:.0f}%", ha="center", va="center", color="white",
                         fontsize=FONT_SIZES["legend.fontsize"])
        left += frac
    ax2.set_yticks(y, names)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Predicted CDS (%)")
    ax2.set_title("B  Predicted CDS lengths", loc="left")
    ax2.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.06), handlelength=1.2,
               columnspacing=1.2)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}/length_distributions.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")


if __name__ == "__main__":
    main()
