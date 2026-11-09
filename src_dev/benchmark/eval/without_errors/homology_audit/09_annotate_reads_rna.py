"""Step 9: mark the test reads that overlap an annotated rRNA or tRNA gene.

Each read is placed on the forward strand of its test genome (complement-strand reads are stored in reverse-complement
coordinates: forward start = L - start_coord - read_length + 2) and compared with the rRNA and tRNA features of the
genome's NCBI GFF, on either strand. A gene counts if it overlaps the read by > 60 bp, the minimum length of a labeled
CDS fragment; rRNA takes precedence over tRNA.

Output: $RESULTS/read_rna_annotation.tsv.gz (genome, read_name, rna_gene: "rRNA", "tRNA" or empty)
Usage: python 09_annotate_reads_rna.py [--work ...] [--results ...] [--procs 48]
"""
import argparse
import bisect
from multiprocessing import Pool

import pandas as pd

GENOMES = "/tmp/nrt204/FragmentPredictor/data/raw_data/genome_data"
MIN_OVERLAP = 61
MAX_GENE_LENGTH = 10_000   # longest rRNA genes are ~3 kb; bounds the backwards search below


def load_genes(acc):
    """{contig: (starts, ends, types)} of the rRNA and tRNA genes, sorted by start; contig lengths."""
    genes, lengths = {}, {}
    with open(f"{GENOMES}/{acc}/genomic.gff") as fh:
        for line in fh:
            if line.startswith("##sequence-region"):
                _, seqid, _, end = line.split()[:4]
                lengths[seqid] = int(end)
            elif line[0] != "#":
                c = line.split("\t")
                if len(c) > 4 and c[2] in ("rRNA", "tRNA"):
                    genes.setdefault(c[0], []).append((int(c[3]), int(c[4]), c[2]))
    return {k: tuple(map(list, zip(*sorted(v)))) for k, v in genes.items()}, lengths


def annotate(task):
    acc, work = task
    genes, lengths = load_genes(acc)
    reads = pd.read_csv(f"{work}/staged/test_csv/{acc}.csv.gz", usecols=["read_name", "read", "start_coord", "assembly_id"])
    rows = []
    for name, read, s, contig in zip(reads["read_name"], reads["read"], reads["start_coord"], reads["assembly_id"]):
        n = len(read)
        if "complement" in name:
            s = lengths[contig] - s - n + 2
        e = s + n - 1
        found = set()
        if contig in genes:
            starts, ends, types = genes[contig]
            i = bisect.bisect_right(starts, e) - 1
            while i >= 0 and starts[i] >= s - MAX_GENE_LENGTH:
                if min(e, ends[i]) - max(s, starts[i]) + 1 >= MIN_OVERLAP:
                    found.add(types[i])
                i -= 1
        rows.append((acc, name, "rRNA" if "rRNA" in found else "tRNA" if "tRNA" in found else ""))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/homology_audit_work")
    parser.add_argument("--results", default="/tmp/nrt204/FragmentPredictor/data/processed_data/homology_audit")
    parser.add_argument("--procs", type=int, default=48)
    args = parser.parse_args()
    accs = pd.read_csv(f"{args.results}/per_read_nt.tsv.gz", sep="\t", usecols=["genome"])["genome"].unique()
    with Pool(args.procs) as pool:
        rows = [r for res in pool.map(annotate, [(a, args.work) for a in accs]) for r in res]
    out = pd.DataFrame(rows, columns=["genome", "read_name", "rna_gene"])
    out.to_csv(f"{args.results}/read_rna_annotation.tsv.gz", sep="\t", index=False)
    print(out["rna_gene"].replace("", "none").value_counts().to_string())
    print("STEP 9 DONE")


if __name__ == "__main__":
    main()
