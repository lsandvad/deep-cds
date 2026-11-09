"""Step 3: assign each test fragment to an identity bin from its best MMseqs2 hit.

Best hit = highest bitscore (ties: first in MMseqs2 order); its fident defines the bin:
no hit, <30%, 30-50%, 50-70%, 70-90%, >=90%. "No hit" = no alignment of >= 20 aa with E <= 10 to any train/val fragment.

Output: $WORK/bins.tsv.gz (one row per test fragment)
Usage: python 03_assign_bins.py [--work /tmp/nrt204/homology_audit_work]
"""
import argparse

import numpy as np
import pandas as pd

BIN_ORDER = ["no hit", "<30%", "30-50%", "50-70%", "70-90%", ">=90%"]
BIN_EDGES = [0.0, 0.3, 0.5, 0.7, 0.9]   # left-closed: [0.3, 0.5) -> "30-50%", ..., [0.9, 1.0] -> ">=90%"
M8_COLS = ["query", "target", "fident", "alnlen", "qlen", "tlen", "evalue", "bits", "qstart", "qend"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/homology_audit_work")
    args = parser.parse_args()

    frag = pd.read_csv(f"{args.work}/fragments/query_fragments.tsv.gz", sep="\t")
    hits = pd.read_csv(f"{args.work}/mmseqs_e10/result.m8.gz", sep="\t", names=M8_COLS, engine="pyarrow")
    hits["order"] = np.arange(len(hits))

    # the top-10 cut in step 2 assumes MMseqs2 lists the max-bitscore hit first
    first = hits.groupby("query", sort=False)["bits"].first()
    print(f"queries whose first hit is not the max-bitscore hit: {(first < hits.groupby('query', sort=False)['bits'].max()).sum():,}")

    best = (hits.sort_values(["query", "bits", "order"], ascending=[True, False, True])
                .drop_duplicates("query").set_index("query"))
    b = best.reindex(frag["fragment_id"])
    assert (b["qlen"].dropna().to_numpy() == frag.loc[b["qlen"].notna().to_numpy(), "aa_len"].to_numpy()).all()
    for col in ["fident", "alnlen", "evalue", "bits"]:
        frag[col] = b[col].to_numpy()
    idx = np.searchsorted(BIN_EDGES, frag["fident"].fillna(0).to_numpy(), side="right")
    frag["bin"] = np.where(frag["fident"].notna(), np.array(BIN_ORDER)[idx], "no hit")

    frag.to_csv(f"{args.work}/bins.tsv.gz", sep="\t", index=False)
    print(frag["bin"].value_counts().reindex(BIN_ORDER).to_string())
    print("STEP 3 DONE")


if __name__ == "__main__":
    main()
