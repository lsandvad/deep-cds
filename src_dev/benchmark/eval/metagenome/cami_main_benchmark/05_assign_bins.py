"""Step 5: identity bin per CAMI fragment and aggregated counts for the results notebook.

Best hit = max bitscore among the top-10 hits (ties: MMseqs2 order); bins on its fident as in the homology
audit: no hit, <30%, 30-50%, 50-70%, 70-90%, >=90% (left-closed). Fragments < 20 aa were not searched
and count as "no hit".

Output: $RESULTS/cami_fragment_counts.tsv - per genome_bin_id x identity bin x length class x model:
n (labeled fragments) and tp. Per-fragment table: $WORK/fragments/fragments_binned.tsv.gz
Usage: python 05_assign_bins.py [--work /tmp/nrt204/cami_overlap_work]
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = "/tmp/nrt204/FragmentPredictor/data/processed_data/cami_main_benchmark"
BIN_ORDER = ["no hit", "<30%", "30-50%", "50-70%", "70-90%", ">=90%"]
BIN_EDGES = [0.0, 0.3, 0.5, 0.7, 0.9]
LENGTH_BINS, LENGTH_LABELS = [0, 29, 39, 49, 10_000], ["20-29 aa", "30-39 aa", "40-49 aa", ">=50 aa"]
MODELS = ["fgs_complete", "fgs_illumina_5", "fgs_illumina_10", "prodigal",
          "deep_cds_no_errors", "deep_cds_substitution", "deep_cds_errors"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/cami_overlap_work")
    args = parser.parse_args()

    frag = pd.read_csv(f"{args.work}/fragments/fragments.tsv.gz", sep="\t")
    hits = pd.read_csv(f"{args.work}/mmseqs/result.m8.gz", sep="\t", engine="pyarrow",
                       names=["query", "target", "fident", "alnlen", "qlen", "tlen", "evalue", "bits"])
    hits["order"] = np.arange(len(hits))
    first = hits.groupby("query", sort=False)["bits"].first()
    n_viol = int((first < hits.groupby("query", sort=False)["bits"].max()).sum())
    best = (hits.sort_values(["query", "bits", "order"], ascending=[True, False, True])
            .drop_duplicates("query").set_index("query"))
    print(f"queries with a hit: {len(best):,} of {len(frag):,} fragments; ordering violations: {n_viol}")

    b = best.reindex(frag["fragment_id"])
    frag["fident"] = b["fident"].values
    frag["alnlen"] = b["alnlen"].values
    frag["bin"] = np.where(b["fident"].notna().values,
                           np.array(BIN_ORDER)[np.searchsorted(BIN_EDGES, b["fident"].fillna(0).values, side="right")],
                           "no hit")
    frag["length_class"] = pd.cut(frag["aa_len"], LENGTH_BINS, labels=LENGTH_LABELS).astype(str)
    frag.to_csv(f"{args.work}/fragments/fragments_binned.tsv.gz", sep="\t", index=False, compression="gzip")

    long = frag.melt(id_vars=["genome_bin_id", "domain", "bin", "length_class"], value_vars=MODELS,
                     var_name="model", value_name="tp")
    agg = (long.groupby(["model", "genome_bin_id", "domain", "bin", "length_class"], observed=True)["tp"]
           .agg(n="size", tp="sum").reset_index())
    agg.to_csv(f"{RESULTS}/cami_fragment_counts.tsv", sep="\t", index=False)
    print(frag["bin"].value_counts().reindex(BIN_ORDER).to_string())
    print("STEP 5 DONE")


if __name__ == "__main__":
    main()
