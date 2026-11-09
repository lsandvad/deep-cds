"""Step 7: best train/val read hit (max bitscore) per test read.

Test reads without a hit (E <= 10, >= 60 nt) have empty best-hit columns.
Output: $RESULTS/per_read_nt.tsv.gz (genome, read_name, best_fident, best_alnlen, best_evalue, best_bits)
Usage: python 07_assign_reads_nt.py [--work ...] [--results ...]
"""
import argparse

import pandas as pd

COLS = ["query", "target", "fident", "alnlen", "qstart", "qend", "qlen", "evalue", "bits"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/homology_audit_work")
    parser.add_argument("--results", default="/tmp/nrt204/FragmentPredictor/data/processed_data/homology_audit")
    args = parser.parse_args()

    hits = pd.read_csv(f"{args.work}/nt/search/result.m8.gz", sep="\t", names=COLS,
                       usecols=["query", "fident", "alnlen", "evalue", "bits"])
    best = hits.sort_values(["query", "bits"], ascending=[True, False]).drop_duplicates("query").set_index("query")

    with open(f"{args.work}/nt/query.fna") as f:
        ids = pd.Series([line[1:].rstrip("\n") for line in f if line.startswith(">")])
    out = pd.DataFrame({"genome": ids.str.split("|", n=1).str[0], "read_name": ids.str.split("|", n=1).str[1]})
    out = out.join(best.add_prefix("best_").reindex(ids).reset_index(drop=True))
    out.to_csv(f"{args.results}/per_read_nt.tsv.gz", sep="\t", index=False)
    print(f"{len(out):,} test reads, {out['best_fident'].notna().mean():.1%} with a hit")
    print("STEP 7 DONE")


if __name__ == "__main__":
    main()
