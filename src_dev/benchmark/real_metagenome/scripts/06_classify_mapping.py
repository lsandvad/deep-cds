"""Step 3: assign every read >= 61 bp to a UHGG mapping category from its primary Bowtie2 alignment.

  identity = matching alignment columns / alignment columns (M + I + D), with mismatches + indel bases = NM
  coverage = read bases inside the alignment (M + I, i.e. not soft-clipped) / read length
  mappable      identity >= 0.95 and coverage >= 0.90
  unmappable    no alignment
  intermediate  everything else
Output: WORK/mapping/read_categories.tsv.gz (read_id, category, identity, coverage, species_rep, contig, pos)
        WORK/mapping/category_counts.tsv
Usage: python 06_classify_mapping.py [--work /tmp/nrt204/real_metagenome_poc_work]
"""
import argparse
import gzip
from collections import Counter

import pysam

MIN_ID, MIN_COV = 0.95, 0.90
M, I, D, EQ, X = 0, 1, 2, 7, 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/real_metagenome_poc_work")
    args = parser.parse_args()
    counts = Counter()
    with pysam.AlignmentFile(f"{args.work}/mapping/uhgg.bam", "rb", threads=8) as bam, \
            gzip.open(f"{args.work}/mapping/read_categories.tsv.gz", "wt", compresslevel=3) as out:
        out.write("read_id\tcategory\tidentity\tcoverage\tspecies_rep\tcontig\tpos\n")
        for r in bam:
            if r.is_secondary or r.is_supplementary:
                continue
            if r.is_unmapped:
                counts["unmappable"] += 1
                out.write(f"{r.query_name}\tunmappable\t\t\t\t\t\n")
                continue
            ops = Counter()
            for op, n in r.cigartuples:
                ops[op] += n
            cols = ops[M] + ops[EQ] + ops[X] + ops[I] + ops[D]
            identity = (cols - r.get_tag("NM")) / cols
            coverage = (ops[M] + ops[EQ] + ops[X] + ops[I]) / r.infer_read_length()
            cat = "mappable" if identity >= MIN_ID and coverage >= MIN_COV else "intermediate"
            counts[cat] += 1
            species, contig = r.reference_name.split("|", 1)
            out.write(f"{r.query_name}\t{cat}\t{identity:.4f}\t{coverage:.4f}\t{species}\t{contig}\t{r.reference_start + 1}\n")
    total = sum(counts.values())
    with open(f"{args.work}/mapping/category_counts.tsv", "w") as f:
        f.write("category\treads\tfraction\n")
        for cat in ("mappable", "intermediate", "unmappable"):
            f.write(f"{cat}\t{counts[cat]}\t{counts[cat] / total:.4f}\n")
    print(open(f"{args.work}/mapping/category_counts.tsv").read(), f"total {total}")


if __name__ == "__main__":
    main()
