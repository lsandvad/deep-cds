"""How often do annotated CDSs share a stop codon but differ in start, so that the preprocessing keeps
the longest one?

The preprocessing (src_dev/data_preprocessing/process_reads_without_indels.py: extract_CDS +
quality_check_cds_annotations) pools the certain RefSeq and GenBank CDS annotations per assembly and strand, removes
duplicates, resolves same-start conflicts (keeps the longest, i.e. latest stop), then resolves same-stop conflicts: the
smallest start (longest CDS) is kept and the largest start dropped. This script rebuilds the same pooled lists, but
remembers which annotation source(s) each coordinate set came from, and records every same-stop group.

Check: for every genome, the final certain CDS sets from quality_check_cds_annotations on the rebuilt lists must equal
those of the original extract_CDS. The one accepted exception is a bug in convert_complement_coordinates: an assembly
without certain complement-strand CDSs inherits the previous assembly's lists, converted to negative coordinates
(never overlapping a read). Such assemblies are counted (n_assemblies_negative_coords) and skipped in the check.

Outputs (RES): start_conflict_groups.tsv (one row per same-stop group), start_conflicts_per_genome.tsv
Usage: python 01_count_start_conflicts.py [--procs 64]
"""
import argparse
import copy
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

from Bio import SeqIO

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "/tmp/nrt204/FragmentPredictor/data/processed_data/start_annotation_audit"
sys.path.insert(0, f"{HERE}/..")
import process_reads_without_indels as prep  # noqa: E402

BASE = prep.data_base_path
UNCERTAIN_TAGS = ("pseudo=true", "product=hypothetical protein", "partial=true", "ab initio prediction",
                  "note=programmed frameshift")


def parse_gff(path, n_header, conv):
    """Certain + strand / - strand CDS coordinates per assembly, with the same filters as extract_CDS."""
    certain = {"+": defaultdict(list), "-": defaultdict(list)}
    uncertain = {"+": defaultdict(list), "-": defaultdict(list)}
    with open(path) as fh:
        for _ in range(n_header):
            next(fh)
        for line in fh:
            if "\tCDS\t" not in line:
                continue
            strand = "+" if "\t+\t" in line else "-" if "\t-\t" in line else None
            if strand is None:
                continue
            f = line.split("\t")
            asm = conv(f[0])
            low = line.lower()
            coords = [int(f[3]), int(f[4])]
            if any(t in low for t in UNCERTAIN_TAGS):
                uncertain[strand][asm].append(coords)
            elif "pseudogene" not in low:
                certain[strand][asm].append(coords)
    return certain, uncertain


def genome(acc):
    gb = acc.replace("GCF_", "GCA_")
    details = prep.get_assembly_details(acc, {})
    try:
        details = prep.get_assembly_details(gb, details)
    except FileNotFoundError:
        pass
    conversion = {v[-1]: v[0] for v in details.values()}

    sources = {}
    gb_gff = f"{BASE}/data/raw_data/genome_data/{gb}/genomic.gff"
    if os.path.exists(gb_gff):
        sources["GenBank"] = parse_gff(gb_gff, 7, lambda a: conversion[a])
    rs_gff = f"{BASE}/data/raw_data/genome_data/{acc}/genomic.gff"
    if os.path.exists(rs_gff):
        sources["RefSeq"] = parse_gff(rs_gff, 9, lambda a: a)

    # Sequence lengths for the complement-strand coordinate conversion (RefSeq FASTA, as convert_complement_coordinates)
    gdir = f"{BASE}/data/raw_data/genome_data/{acc}"
    fasta = [f for f in os.listdir(gdir) if f.startswith(acc)][0]
    seq_len = {r.id: len(r.seq) for r in SeqIO.parse(f"{gdir}/{fasta}", "fasta")}

    def to_rc(coords, asm):
        return [[seq_len[asm] - e + 1, seq_len[asm] - s + 1] for s, e in coords]

    # Pooled lists exactly as extract_CDS builds them (GenBank first, then RefSeq), and the source of each coordinate set
    pooled = {"+": defaultdict(list), "-": defaultdict(list)}
    pooled_unc = {"+": defaultdict(list), "-": defaultdict(list)}
    origin = defaultdict(set)  # (strand, asm, start, end) -> {"RefSeq", "GenBank"}
    for src, (cert, unc) in sources.items():
        for strand in "+-":
            for asm, cs in cert[strand].items():
                if strand == "-":
                    cs = to_rc(cs, asm)
                pooled[strand][asm] += cs
                for s, e in cs:
                    origin[(strand, asm, s, e)].add(src)
            for asm, cs in unc[strand].items():
                pooled_unc[strand][asm] += to_rc(cs, asm) if strand == "-" else cs

    # Check against the original preprocessing
    orig = prep.extract_CDS(acc)
    negative = set()
    for strand, (ci, ui) in (("+", (0, 1)), ("-", (2, 3))):
        cert = {a: list(v) for a, v in pooled[strand].items()}
        unc = {a: list(v) for a, v in pooled_unc[strand].items()}
        for a in set(cert) | set(unc):
            cert.setdefault(a, [])
            unc.setdefault(a, [])
        mine, _ = prep.quality_check_cds_annotations(copy.deepcopy(cert), copy.deepcopy(unc))
        o = orig[ci]
        for a in set(mine) | set(o):
            if sorted(map(tuple, mine.get(a, []))) != sorted(map(tuple, o.get(a, []))):
                if strand == "-" and any(c[0] < 1 for c in o.get(a, [])):
                    negative.add(a)
                    continue
                raise RuntimeError(f"{acc} {strand} {a}: rebuilt CDS set differs from extract_CDS")
    final = {"+": orig[0], "-": orig[2]}

    # Same-stop groups, after duplicate removal and the same-start (selenocysteine) step, as in the original
    groups = []
    for strand in "+-":
        for asm, cs in pooled[strand].items():
            cs = sorted(set(map(tuple, cs)))
            by_start = defaultdict(list)
            for s, e in cs:
                by_start[s].append(e)
            for s, ends in by_start.items():
                if len(ends) > 1:
                    cs.remove((s, min(ends)))
            by_end = defaultdict(list)
            for s, e in cs:
                by_end[e].append(s)
            for e, starts in by_end.items():
                if len(starts) < 2:
                    continue
                starts = sorted(starts)
                src = {s: origin[(strand, asm, s, e)] for s in starts}
                rs = [s for s in starts if "RefSeq" in src[s]]
                gbs = [s for s in starts if "GenBank" in src[s]]
                if len(rs) == 1 and len(gbs) == 1:
                    kind = "RefSeq vs GenBank"
                elif len(rs) > 1 or len(gbs) > 1:
                    kind = "several starts in one source"
                else:
                    kind = "other"
                kept = starts[0]
                groups.append({"accession": acc, "assembly": asm, "strand": strand, "stop_end": e,
                               "n_starts": len(starts), "starts": ",".join(map(str, starts)),
                               "kept_start": kept, "dropped_start": starts[-1],
                               "offset_codons": (starts[-1] - kept) / 3,
                               "kept_sources": "+".join(sorted(src[kept])),
                               "dropped_sources": "+".join(sorted(src[starts[-1]])), "kind": kind,
                               "kept_in_final": [kept, e] in final[strand].get(asm, [])})
    n_final = sum(len(v) for st in final.values() for v in st.values())
    return acc, n_final, groups, len(negative)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--procs", type=int, default=64)
    p.add_argument("--partitions", nargs="+", default=["test"], choices=["train", "val", "test"])
    a = p.parse_args()
    parts = {}
    for part in a.partitions:
        for acc in open(f"{BASE}/data/processed_data/genome_partitions/{part}_partition_accessions.txt").read().split():
            parts[acc] = part
    os.makedirs(RES, exist_ok=True)
    with Pool(a.procs) as pool:
        res = pool.map(genome, sorted(parts), chunksize=1)

    cols = ["partition", "accession", "assembly", "strand", "stop_end", "n_starts", "starts", "kept_start",
            "dropped_start", "offset_codons", "kept_sources", "dropped_sources", "kind", "kept_in_final"]
    with open(f"{RES}/start_conflict_groups.tsv", "w") as g, open(f"{RES}/start_conflicts_per_genome.tsv", "w") as s:
        g.write("\t".join(cols) + "\n")
        s.write("partition\taccession\tn_certain_cds\tn_same_stop_groups\tn_assemblies_negative_coords\n")
        for acc, n_final, groups, n_neg in res:
            s.write(f"{parts[acc]}\t{acc}\t{n_final}\t{len(groups)}\t{n_neg}\n")
            for r in groups:
                g.write("\t".join(str(({"partition": parts[acc]} | r)[c]) for c in cols) + "\n")
    print(f"{len(res)} genomes; rebuilt CDS sets identical to extract_CDS for all except "
          f"{sum(r[3] for r in res)} assemblies with negative coordinates (convert_complement_coordinates bug)")


if __name__ == "__main__":
    main()
