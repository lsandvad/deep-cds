"""Step 10: share of each genome covered by annotated rRNA and tRNA genes.

For every genome of the dataset (train, val, test), the rRNA and tRNA genes of its NCBI GFF are merged into
non-overlapping intervals (both strands together) and the covered positions are divided by the genome length
(sum of the ##sequence-region lengths). Genome statistics only, so the 3 translation table 4 test genomes are included.

Output: $RESULTS/rna_gene_coverage_per_genome.tsv (genome, partition, genome_length, rRNA_bp, tRNA_bp, rna_bp and
        the same as fractions of the genome length)
Usage: python 10_rna_gene_coverage.py [--results ...] [--procs 64]
"""
import argparse
from multiprocessing import Pool

import pandas as pd

DATA = "/tmp/nrt204/FragmentPredictor/data"


def covered(intervals):
    """Number of positions covered by closed (start, end) intervals."""
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e + 1:
            if cur_e is not None:
                total += cur_e - cur_s + 1
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s + 1 if cur_e is not None else 0)


def genome(task):
    acc, partition = task
    length, genes = 0, {"rRNA": {}, "tRNA": {}}
    with open(f"{DATA}/raw_data/genome_data/{acc}/genomic.gff") as fh:
        for line in fh:
            if line.startswith("##sequence-region"):
                length += int(line.split()[3])
            elif line[0] != "#":
                c = line.split("\t")
                if len(c) > 4 and c[2] in genes:
                    genes[c[2]].setdefault(c[0], []).append((int(c[3]), int(c[4])))
    row = {"genome": acc, "partition": partition, "genome_length": length}
    for t in genes:
        row[f"{t}_bp"] = sum(covered(v) for v in genes[t].values())
    contigs = set(genes["rRNA"]) | set(genes["tRNA"])
    row["rna_bp"] = sum(covered(genes["rRNA"].get(c, []) + genes["tRNA"].get(c, [])) for c in contigs)
    for k in ("rRNA", "tRNA", "rna"):
        row[f"{k}_fraction"] = row[f"{k}_bp"] / length
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=f"{DATA}/processed_data/homology_audit")
    parser.add_argument("--procs", type=int, default=64)
    args = parser.parse_args()
    tasks = []
    for part in ("train", "val", "test"):
        with open(f"{DATA}/processed_data/genome_partitions/{part}_partition_accessions.txt") as f:
            tasks += [(acc, part) for acc in f.read().split()]
    with Pool(args.procs) as pool:
        out = pd.DataFrame(pool.map(genome, tasks, chunksize=1))
    out.to_csv(f"{args.results}/rna_gene_coverage_per_genome.tsv", sep="\t", index=False)
    print(f"{len(out)} genomes, median rRNA/tRNA share {100 * out['rna_fraction'].median():.2f}%")
    print("STEP 10 DONE")


if __name__ == "__main__":
    main()
