"""Sanity checks / spot checks.

1. 10 random reads per UHGG mapping category: primary alignment (CIGAR, NM), identity, coverage, category.
2. Predicted CDS >= 61 bp on 2,000 random reads: translated (table 11; MetaProdigal: the table of its model for the
   read) from the read sequence in the tool's
   frame; reports the fraction of CDS with an internal stop codon (expected ~0 for every tool).
Output: RES/spot_check_mapping.tsv, RES/spot_check_translation.tsv (and printed)
Usage: python 09_spot_checks.py [--work ...] [--deepcds_dir ...]
"""
import argparse
import glob
import gzip
import os
import random
import subprocess

import pandas as pd
import pysam

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "/tmp/nrt204/FragmentPredictor/real_metagenome_poc/results"
SEED, N_PER_CAT, N_TRANSL = 42, 10, 2000
CKPT = "full_model_all_genomes_seed_42_trained_final_8M_no_dropout"
COMP = str.maketrans("ACGTN", "TGCAN")
BASES = "TCAG"
CODE11 = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
TABLE = {a + b + c: CODE11[16 * i + 4 * j + k] for i, a in enumerate(BASES) for j, b in enumerate(BASES)
         for k, c in enumerate(BASES)}


TABLE4 = {**TABLE, "TGA": "W"}  # MetaProdigal's meta mode uses translation-table-4 models for some reads


def translate(nt, table=11):
    t = TABLE4 if table == 4 else TABLE
    return "".join(t.get(nt[i:i + 3], "X") for i in range(0, len(nt) - 2, 3))


def load_reads(work, wanted):
    seqs = {}
    proc = subprocess.Popen(["pigz", "-dc", f"{work}/fasta/reads_ge61.fa.gz"], stdout=subprocess.PIPE, text=True)
    name = None
    for line in proc.stdout:
        if line[0] == ">":
            name = line[1:].rstrip("\n")
        elif name in wanted:
            seqs[name] = line.rstrip("\n")
    proc.wait()
    return seqs


def cds_from_gff(paths, wanted, deepcds=False, tables=None):
    """{read: [(start, end, strand)]} for CDS >= 61 bp; DeepCDS _rc entries are mapped back to the read.
    tables (Prodigal GFF only): filled with the genetic code of the MetaProdigal model chosen for each read."""
    out = {}
    for path in paths:
        with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as fh:
            seqhdr = None
            for line in fh:
                if line.startswith("#"):
                    if tables is not None:
                        if line.startswith("# Sequence Data:"):
                            seqhdr = line.split('seqhdr="', 1)[1].split('"', 1)[0]
                        elif line.startswith("# Model Data:") and seqhdr in wanted:
                            tables[seqhdr] = int(line.split("transl_table=", 1)[1].split(";", 1)[0])
                    continue
                c = line.split("\t", 9)
                if len(c) < 9 or c[2] != "CDS":
                    continue
                read, strand = c[0], c[6]
                if deepcds:
                    if "group_id=" in c[8]:
                        continue  # indel-split segments: frame shifts inside the CDS, not a clean translation
                    if read.endswith("_rc"):
                        read, strand = read[:-3], "rc"
                if read in wanted and int(c[4]) - int(c[3]) + 1 >= 61:
                    out.setdefault(read, []).append((int(c[3]), int(c[4]), strand))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/real_metagenome_poc_work")
    parser.add_argument("--deepcds_dir", default="/tmp/nrt204/real_metagenome_poc_work/preds/DeepCDS")
    args = parser.parse_args()
    rnd = random.Random(SEED)

    # 1. mapping
    cats = pd.read_csv(f"{args.work}/mapping/read_categories.tsv.gz", sep="\t")
    pick = pd.concat([cats[cats["category"] == c].sample(N_PER_CAT, random_state=SEED)
                      for c in ("mappable", "intermediate", "unmappable")])
    wanted = set(pick["read_id"])
    info = {}
    with pysam.AlignmentFile(f"{args.work}/mapping/uhgg.bam", "rb", threads=8) as bam:
        for r in bam:
            if r.query_name in wanted:
                info[r.query_name] = (r.cigarstring or "*", r.get_tag("NM") if r.has_tag("NM") else "",
                                      r.infer_read_length())
                if len(info) == len(wanted):
                    break
    pick["cigar"], pick["NM"], pick["read_len"] = zip(*(info[r] for r in pick["read_id"]))
    pick.to_csv(f"{RES}/spot_check_mapping.tsv", sep="\t", index=False)
    print(pick.to_string(index=False))

    # 2. translation
    ids = list(cats["read_id"])
    wanted = set(rnd.sample(ids, N_TRANSL))
    seqs = load_reads(args.work, wanted)
    sources = {"prodigal": (sorted(glob.glob(f"{args.work}/preds/prodigal/chunk_*.gff.gz")), False)}
    for m in ("complete", "illumina_5", "illumina_10"):
        sources[f"fgs_{m}"] = (sorted(glob.glob(f"{args.work}/preds/fgs_{m}/chunk_*.gff.gz")), False)
    for t, d in {"deepcds_S": "model_with_substitution_errors", "deepcds_SI": "model_with_errors"}.items():
        p = f"{args.deepcds_dir}/{d}/real_metagenome_poc/{CKPT}/predictions_reads_all.gff"
        sources[t] = ([p] if os.path.exists(p) else [], True)
    rows = []
    for tool, (paths, deep) in sources.items():
        if not paths:
            continue
        tables = {} if tool == "prodigal" else None
        cds = cds_from_gff(paths, wanted, deepcds=deep, tables=tables)
        n = internal = 0
        for read, items in cds.items():
            seq = seqs[read]
            for s, e, strand in items:
                if strand in ("-", "rc"):
                    full = seq.translate(COMP)[::-1]
                    nt = full[s - 1:e] if strand == "rc" else full[len(seq) - e:len(seq) - s + 1]
                else:
                    nt = seq[s - 1:e]
                aa = translate(nt, tables.get(read, 11) if tables is not None else 11)
                n += 1
                internal += "*" in aa[:-1]
        rows.append({"tool": tool, "reads_sampled": len(wanted), "cds_ge61": n, "with_internal_stop": internal,
                     "fraction": internal / n if n else float("nan")})
    t = pd.DataFrame(rows)
    t.to_csv(f"{RES}/spot_check_translation.tsv", sep="\t", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
