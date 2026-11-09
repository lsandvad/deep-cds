"""Step 1: extract and translate the labeled CDS fragments.

Query  = every labeled CDS fragment (> 60 bp) of the error-free 300 bp test set, i.e. every entry of testset_dict.pkl,
         the units the CDS-level benchmark counts as TP or FN.
Target = every labeled CDS fragment (>= 20 aa) of the error-free train/val reads, the data DeepCDS N was trained on.

Coordinates are 1-based inclusive [start, end, rf] on the read. Each fragment is translated in its labeled frame
(complete codons only, terminal stop removed) with NCBI table 4 for the genomes in table4_accessions, table 11 otherwise.
Identical target proteins are written once.

Outputs ($WORK/fragments/): query.fasta, query_fragments.tsv.gz (one row per test fragment), target.fasta
Usage: python 01_extract_fragments.py [--work /tmp/nrt204/homology_audit_work] [--procs 64]
"""
import argparse
import ast
import os
import pickle
import re
from multiprocessing import Pool

import pandas as pd
from Bio.Seq import Seq

MIN_AA = 20


def read_accessions(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def translate_fragment(read_seq, start, end, rf, table):
    """Protein of read[start-1:end] in its labeled frame; internal stops (should not occur) become X."""
    assert (start - 1) % 3 == int(rf), f"frame mismatch: start={start} rf={rf}"
    nt = read_seq[start - 1:end]
    aa = str(Seq(nt[:len(nt) // 3 * 3]).translate(table=table))
    return aa[:-1].replace("*", "X") if aa.endswith("*") else aa.replace("*", "X")


def test_genome(args):
    acc, staged, table, family = args
    with open(f"{staged}/testset/{acc}/testset_dict.pkl", "rb") as f:
        testset = pickle.load(f)
    reads = pd.read_csv(f"{staged}/test_csv/{acc}.csv.gz", usecols=["read_name", "read"])
    seqs = dict(zip(reads["read_name"], reads["read"]))
    rows = []
    for read_name, entry in testset.items():
        # error-free reads: no CDS is split by an indel, so coord_idx indexes the list the benchmark matches on
        assert all(len(c) == 1 for c in entry["cds_fragments_connection"]), read_name
        for coord_idx, (start, end, rf) in enumerate(entry["cds_coords"]):
            aa = translate_fragment(seqs[read_name], start, end, rf, table)
            rows.append((acc, family, table, read_name, coord_idx, start, end, int(rf), len(aa), aa))
    return rows


def train_val_genome(args):
    acc, staged, table = args
    reads = pd.read_csv(f"{staged}/train_val_csv/{acc}.csv.gz", usecols=["read", "cds_coords"])
    proteins = set()
    for read_seq, coords in zip(reads["read"], reads["cds_coords"]):
        for start, end, rf in ast.literal_eval(coords):
            aa = translate_fragment(read_seq, start, end, rf, table)
            if len(aa) >= MIN_AA:
                proteins.add(aa)
    return proteins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/homology_audit_work")
    parser.add_argument("--procs", type=int, default=64)
    args = parser.parse_args()
    staged, out = f"{args.work}/staged", f"{args.work}/fragments"
    os.makedirs(out, exist_ok=True)

    with open(f"{staged}/dataset_information/table4_accessions") as f:
        table4 = set(re.findall(r'"(GCF_\d+\.\d+)"', f.read()))
    family = pd.read_csv(f"{staged}/dataset_information/genomes_info_with_partitions.csv").set_index("accession")["family"]
    test = read_accessions(f"{staged}/genome_partitions/test_partition_accessions.txt")
    train_val = (read_accessions(f"{staged}/genome_partitions/train_partition_accessions.txt")
                 + read_accessions(f"{staged}/genome_partitions/val_partition_accessions.txt"))
    tt = lambda acc: 4 if acc in table4 else 11

    with Pool(args.procs) as pool:
        q = [r for rows in pool.map(test_genome, [(a, staged, tt(a), family[a]) for a in test], chunksize=1) for r in rows]
        target = set().union(*pool.imap_unordered(train_val_genome, [(a, staged, tt(a)) for a in train_val], chunksize=1))

    q = pd.DataFrame(q, columns=["genome", "family", "translation_table", "read_name", "coord_idx", "start", "end", "rf",
                                 "aa_len", "protein"])
    q.insert(0, "fragment_id", [f"q{i:08d}" for i in range(len(q))])
    assert (q["aa_len"] >= MIN_AA).all()   # > 60 bp always gives >= 20 codons
    with open(f"{out}/query.fasta", "w") as f:
        f.writelines(f">{i}\n{aa}\n" for i, aa in zip(q["fragment_id"], q["protein"]))
    q.drop(columns="protein").to_csv(f"{out}/query_fragments.tsv.gz", sep="\t", index=False)
    with open(f"{out}/target.fasta", "w") as f:
        f.writelines(f">t{i:09d}\n{aa}\n" for i, aa in enumerate(sorted(target)))
    print(f"{len(q):,} test fragments ({len(test)} genomes); {len(target):,} unique train/val proteins ({len(train_val)} genomes)")
    print("STEP 1 DONE")


if __name__ == "__main__":
    main()
