"""How many test reads contain a start codon that was dropped by the longest-CDS rule?

A dropped start lies inside the kept (longer) CDS, so reads label it as coding (1), not as start (2). A prediction of
a start there is scored as a false-positive start codon: these counts are the maximum number of start-codon FPs the rule
can cause per error-free test set.

For every test genome and read length, reads of read_names_list.pkl (the evaluated reads) are taken from
reads_processed/test/without_errors_{L}bp/csv (start_coord: 1-based, on the read's own strand, the same coordinates
as the step-01 table). A read contains a start if the whole codon lies inside the read. Check: the read's label at a
dropped start must be 1, at a kept start 2.

Outputs (RES): affected_test_reads.tsv (per read length and conflict kind)
Usage: python 02_count_affected_test_reads.py [--procs 64]
"""
import argparse
import ast
import os
import pickle
from collections import Counter, defaultdict
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "/tmp/nrt204/FragmentPredictor/data/processed_data/start_annotation_audit"
BASE = "/tmp/nrt204/FragmentPredictor/data/processed_data"
LENGTHS = [75, 100, 150, 300, 700, 1000]
CONFLICTS = None


def init(conflicts):
    global CONFLICTS
    CONFLICTS = conflicts


def genome_length(task):
    acc, length = task
    names = set(pickle.load(open(f"{BASE}/testset_processed/without_errors_{length}bp/{acc}/read_names_list.pkl", "rb")))
    df = pd.read_csv(f"{BASE}/reads_processed/test/without_errors_{length}bp/csv/{acc}.csv.gz",
                     usecols=["read_name", "read", "start_coord", "assembly_id", "strand",
                              "rf0_labels", "rf1_labels", "rf2_labels"])
    df = df[df["read_name"].isin(names)]
    counts, checks, codons = Counter(), Counter(), Counter()
    reads_hit = defaultdict(set)
    for (asm, strand), g in df.groupby(["assembly_id", "strand"]):
        for pos, role, kind in CONFLICTS.get((acc, asm, strand), []):
            j = pos - g["start_coord"]  # 0-based offset of the codon's first base in the read
            hit = g[(j >= 0) & (j + 2 <= g["read"].str.len() - 1)]
            for _, r in hit.iterrows():
                o = pos - r["start_coord"]
                lab = ast.literal_eval(r[f"rf{o % 3}_labels"])[o // 3]
                checks[(role, lab)] += 1
                if role == "dropped":
                    counts[kind] += 1
                    reads_hit[kind].add(r["read_name"])
                    codons[r["read"][o:o + 3]] += 1
    return acc, length, len(df), counts, {k: len(v) for k, v in reads_hit.items()}, checks, codons


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--procs", type=int, default=64)
    a = p.parse_args()
    groups = pd.read_csv(f"{RES}/start_conflict_groups.tsv", sep="\t")
    groups = groups[(groups["partition"] == "test") & groups["kept_in_final"]]
    conflicts = defaultdict(list)
    for _, r in groups.iterrows():
        k = (r["accession"], r["assembly"], r["strand"])
        conflicts[k].append((r["kept_start"], "kept", r["kind"]))
        conflicts[k].append((r["dropped_start"], "dropped", r["kind"]))
    accs = open(f"{BASE}/genome_partitions/test_partition_accessions.txt").read().split()
    tasks = [(acc, L) for L in LENGTHS for acc in accs]
    with Pool(a.procs, initializer=init, initargs=(dict(conflicts),)) as pool:
        res = pool.map(genome_length, tasks, chunksize=1)

    rows, checks, codons = [], defaultdict(Counter), defaultdict(Counter)
    for L in LENGTHS:
        rl = [r for r in res if r[1] == L]
        n_reads = sum(r[2] for r in rl)
        for kind in sorted(groups["kind"].unique()):
            rows.append({"read_length": L, "kind": kind, "n_test_reads": n_reads,
                         "dropped_start_occurrences": sum(r[3][kind] for r in rl),
                         "reads_with_dropped_start": sum(r[4].get(kind, 0) for r in rl)})
        for r in rl:
            checks[L] += r[5]
            codons[L] += r[6]
    out = pd.DataFrame(rows)
    out.to_csv(f"{RES}/affected_test_reads.tsv", sep="\t", index=False)
    print(out.to_string(index=False))
    for L in LENGTHS:
        print(f"{L} bp label check (role, label): {dict(checks[L])}; dropped-start codons: {dict(codons[L].most_common(6))}")


if __name__ == "__main__":
    main()
