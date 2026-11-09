"""Step 8: CDS-level TP/FP/FN per test read and tool (exact match, IoU = 1), for the nucleotide-level evaluation.

Same matching as step 4, read by read; reads without labels count their predictions as FP. Per-genome sums are
checked against per_genome_counts.tsv from step 4.

Output: $RESULTS/per_read_counts_nt.tsv.gz (genome, read_name, {tool}_tp/_fp/_fn), only reads with a non-zero count
Usage: python 08_evaluate_reads_nt.py [--work ...] [--results ...] [--procs 48]
"""
import argparse
import os
import pickle
from multiprocessing import Pool

import pandas as pd

import benchmark_functions as bf

TOOLS = ["deepcds_n", "metaprodigal", "fgs_complete"]


def evaluate_genome(args):
    acc, staged = args
    with open(f"{staged}/testset/{acc}/testset_dict.pkl", "rb") as f:
        testset = bf.get_full_cds(pickle.load(f))
    with open(f"{staged}/testset/{acc}/read_names_list.pkl", "rb") as f:
        read_names = pickle.load(f)
    preds = {}
    for tool in TOOLS:
        with open(f"{staged}/preds/{tool}/{acc}.pkl", "rb") as f:
            preds[tool] = bf.get_full_cds(pickle.load(f))
    rows = []
    for read in read_names:
        actual = testset.get(read, {}).get("cds_coords", [])
        row = [acc, read]
        for tool in TOOLS:
            predicted = preds[tool].get(read, {}).get("cds_coords", [])
            _, tp = bf.find_matching_genes(predicted, actual)
            row += [tp, len(predicted) - tp, len(actual) - tp]
        if any(row[2:]):
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/homology_audit_work")
    parser.add_argument("--results", default="/tmp/nrt204/FragmentPredictor/data/processed_data/homology_audit")
    parser.add_argument("--procs", type=int, default=48)
    args = parser.parse_args()
    staged = f"{args.work}/staged"
    with Pool(args.procs) as pool:
        res = pool.map(evaluate_genome, [(a, staged) for a in sorted(os.listdir(f"{staged}/testset"))])
    cols = ["genome", "read_name"] + [f"{t}_{k}" for t in TOOLS for k in ("tp", "fp", "fn")]
    df = pd.DataFrame([r for rows in res for r in rows], columns=cols)

    ref = pd.read_csv(f"{args.results}/per_genome_counts.tsv", sep="\t")
    sums = df.groupby("genome")[cols[2:]].sum()
    for r in ref.itertuples():
        for k in ("tp", "fp", "fn"):
            assert sums.loc[r.genome, f"{r.tool}_{k}"] == getattr(r, k), (r.genome, r.tool, k)
    df.to_csv(f"{args.results}/per_read_counts_nt.tsv.gz", sep="\t", index=False)
    print(f"{len(df):,} reads with a count; per-genome sums equal step 4")
    print("STEP 8 DONE")


if __name__ == "__main__":
    main()
