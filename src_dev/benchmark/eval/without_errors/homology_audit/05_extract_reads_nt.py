"""Step 5: write the test reads and the train/val reads as nucleotide FASTA.

Query  = every read of the error-free 300 bp test set, coding and non-coding.
Target = every error-free train/val read, the data DeepCDS N was trained on.
Read IDs are "<accession>|<read_name>", as read names are only unique within a genome.

Outputs: $WORK/nt/query.fna, $WORK/nt/target.fna
Usage: python 05_extract_reads_nt.py [--work /tmp/nrt204/homology_audit_work] [--procs 48]
"""
import argparse
import os
from multiprocessing import Pool

import pandas as pd


def to_fasta(path):
    acc = os.path.basename(path)[:-len(".csv.gz")]
    df = pd.read_csv(path, usecols=["read_name", "read"])
    return "".join(f">{acc}|{n}\n{s}\n" for n, s in zip(df["read_name"], df["read"]))


def write(folder, out, procs):
    files = sorted(f"{folder}/{f}" for f in os.listdir(folder) if f.endswith(".csv.gz"))
    with open(out, "w") as fh, Pool(procs) as pool:
        for text in pool.imap(to_fasta, files, chunksize=2):
            fh.write(text)
    return len(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/homology_audit_work")
    parser.add_argument("--procs", type=int, default=48)
    args = parser.parse_args()
    os.makedirs(f"{args.work}/nt", exist_ok=True)
    n_test = write(f"{args.work}/staged/test_csv", f"{args.work}/nt/query.fna", args.procs)
    n_train = write(f"{args.work}/staged/train_val_csv", f"{args.work}/nt/target.fna", args.procs)
    assert (n_test, n_train) == (215, 910), (n_test, n_train)
    print("STEP 5 DONE")


if __name__ == "__main__":
    main()
