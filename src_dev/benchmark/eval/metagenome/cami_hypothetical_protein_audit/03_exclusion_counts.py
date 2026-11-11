"""CDS and reads excluded by each filtering criterion of the CAMI III sample_0 test set.

Replays the per-read decision of src_dev/data_preprocessing/process_and_map_CAMI_testset.py (process_bam), with
its own functions, on the raw sample_0 BAMs and cached GFFs, recording every criterion separately:
  1. no GFF for the contig                                 -> read dropped (incl. the 9 plasmid genomes, whose BAM
                                                              contig names carry a copy prefix, e.g. 0NZ_CP029594.1)
  2. overlaps a CDS flagged pseudogene / pseudo=true / partial=true / note=programmed frameshift (read's strand)
                                                           -> read dropped; flags recorded per marker (not exclusive)
  3. fails the labeling quality check (label_read)        -> read dropped
  4. kept; flagged if it overlaps a hypothetical-protein / ab initio CDS (strand-specific, as in process_bam) and
     if its contig is in CAMI_metagenome_no_cds_contigs/sample_0_contigs_without_cds.txt
CDS level: every CDS line in the GFFs of the contigs with reads, per marker (both strands).
Checks: kept reads = 32,680,743; kept + uncertain flag = 1,337,421; kept on contigs without CDS = 4,321,892.
Output: $RESULTS/exclusion_counts_reads.tsv (genome x criterion), $RESULTS/exclusion_counts_cds.tsv (genome x marker)
Usage: python 03_exclusion_counts.py [--procs 48]
"""
import argparse
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pysam

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = "/tmp/nrt204/FragmentPredictor/data/processed_data/cami_hypothetical_protein_audit"
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "data_preprocessing"))
import process_and_map_CAMI_testset as P  # noqa: E402

D = Path("/tmp/nrt204/FragmentPredictor/data")
BAM_DIR = D / "raw_data/CAMI_datasets/sample_0_bam"
GFF_DIR = D / "raw_data/CAMI_datasets/gff_cache"
NO_CDS_CONTIGS = D / "processed_data/CAMI_metagenome_no_cds_contigs/sample_0_contigs_without_cds.txt"
EXCL = {"pseudogene": "pseudogene", "pseudo": "pseudo=true", "partial": "partial=true",
        "frameshift": "note=programmed frameshift"}
TAG = {"hypothetical": "product=hypothetical protein", "ab_initio": "ab initio prediction"}
no_cds_contigs = set()


def marker_intervals(gff_path, length):
    """{strand: {marker: [[start, end], ...]}} in the strand frame used by process_bam, plus CDS-level counts."""
    out = {"+": {m: [] for m in (*EXCL, *TAG)}, "-": {m: [] for m in (*EXCL, *TAG)}}
    cds = Counter()
    with open(gff_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9 or c[2] != "CDS" or c[6] not in "+-":
                continue
            a = c[8].lower()
            s, e = int(c[3]), int(c[4])
            iv = [s, e] if c[6] == "+" else [length - e + 1, length - s + 1]
            flags = {m for m, k in {**EXCL, **TAG}.items() if k in a}
            excluded = bool(flags & set(EXCL))
            cds["cds"] += 1
            cds["any_excluded"] += excluded
            for m in flags:
                cds[m] += 1
                out[c[6]][m].append(iv)
            if not excluded:
                tag = flags & set(TAG)
                cds["kept"] += 1
                cds["kept_tagged"] += bool(tag)
                cds["kept_hypothetical_only"] += tag == {"hypothetical"}
                cds["kept_ab_initio_only"] += tag == {"ab_initio"}
                cds["kept_hypothetical_and_ab_initio"] += tag == {"hypothetical", "ab_initio"}
    return out, cds


def count_bam(bam_path):
    genome = bam_path.stem.split("_", 1)[1]
    reads, cds_counts, contigs = Counter(), Counter(), {}
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam:
            reads["reads"] += 1
            contig = read.reference_name
            if contig not in contigs:
                gff = GFF_DIR / f"{contig}.gff3"
                if gff.exists() and gff.stat().st_size > 0:
                    try:
                        ann = P.parse_gff3_cds(gff)
                        markers, cc = marker_intervals(gff, ann["length"])
                        contigs[contig] = (ann, markers)
                        cds_counts.update(cc)
                    except ValueError:
                        contigs[contig] = None
                else:
                    contigs[contig] = None
            entry = contigs[contig]
            if entry is None:
                reads["no_gff"] += 1
                continue
            ann, markers = entry
            if read.query_sequence is None:
                reads["no_sequence"] += 1
                continue
            cigar = P.normalize_cigar(read.cigarstring)
            if read.is_reverse:
                seq_len, cigar = len(read.query_sequence), P.reverse_cigar(cigar)
                start, strand = ann["length"] - read.reference_end + 1, "-"
            else:
                seq_len, start, strand = len(read.query_sequence), read.reference_start + 1, "+"
            view = ann[strand]
            hit = {m: bool(P.get_position_gene_overlaps(markers[strand][m], start, seq_len)) for m in (*EXCL, *TAG)}
            if P.get_position_gene_overlaps(view["exclude"].get(contig, []), start, seq_len):
                reads["excluded"] += 1
                for m in EXCL:
                    reads[f"excluded_{m}"] += hit[m]
                continue
            cds_overlaps = P.get_position_gene_overlaps(view["cds"].get(contig, []), start, seq_len)
            if not P.label_read(seq_len, cigar, start, cds_overlaps)[3]:
                reads["quality_fail"] += 1
                continue
            reads["kept"] += 1
            tagged = bool(P.get_position_gene_overlaps(view["tag"].get(contig, []), start, seq_len))
            reads["kept_uncertain"] += tagged
            reads["kept_uncertain_hypothetical"] += hit["hypothetical"]
            reads["kept_uncertain_ab_initio"] += hit["ab_initio"]
            reads["kept_with_cds"] += bool(cds_overlaps)
            reads["kept_no_cds_contig"] += contig in no_cds_contigs
    print(f"  {genome}: {reads['reads']:,} reads", flush=True)
    return genome, reads, cds_counts


def main():
    global no_cds_contigs
    parser = argparse.ArgumentParser()
    parser.add_argument("--procs", type=int, default=48)
    args = parser.parse_args()
    no_cds_contigs = set(NO_CDS_CONTIGS.read_text().split())
    bams = sorted(BAM_DIR.glob("*.bam"), key=lambda p: -p.stat().st_size)
    with Pool(args.procs) as pool:
        res = pool.map(count_bam, bams, chunksize=1)
    reads = pd.DataFrame({g: r for g, r, _ in res}).T.fillna(0).astype(int)
    cds = pd.DataFrame({g: c for g, _, c in res}).T.fillna(0).astype(int)
    reads.index.name = cds.index.name = "genome_bin_id"
    tot = reads.sum()
    print(tot.to_string())
    for k, v in {"reads": 33_267_026, "kept": 32_680_743, "kept_uncertain": 1_337_421, "kept_no_cds_contig": 4_321_892}.items():
        print(f"check {k}: {int(tot[k]):,} (expected {v:,}) {'OK' if int(tot[k]) == v else 'MISMATCH'}")
    os.makedirs(f"{RESULTS}", exist_ok=True)
    reads.to_csv(f"{RESULTS}/exclusion_counts_reads.tsv", sep="\t")
    cds.to_csv(f"{RESULTS}/exclusion_counts_cds.tsv", sep="\t")
    print(cds.sum().to_string())


if __name__ == "__main__":
    main()
