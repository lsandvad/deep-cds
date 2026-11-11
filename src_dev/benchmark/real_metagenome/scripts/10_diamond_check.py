"""Step 6 (optional): do predicted CDS translate to known proteins? DIAMOND against UniRef90.

Population: the first --n_reads reads >= 61 bp (reads_ge61.fa.gz order). Detection as in 07_detection_table.py
(>= 1 CDS on either strand; DeepCDS S = D, MetaProdigal = P, FGS (Complete) = F). Groups = the cells of the
UpSet diagram of D / P / F detection with CDS >= 61 bp (disjoint), plus two extra groups:
  all_tools (D P F), deepcds_prodigal (D P), deepcds_fgs (D F), deepcds_only (D)     -> DeepCDS S peptide
  prodigal_fgs (P F), prodigal_only (P)                                              -> MetaProdigal peptide
  fgs_only (F)                                                                       -> FGS (Complete) peptide
  deepcds_short    DeepCDS S only CDS < 61 bp (any status of the other tools)        -> DeepCDS S peptide
  no_tool          no CDS from any of the three tools (any length)                   -> read, blastx (6 frames)
Up to --per_group reads are sampled per group (seed 42); the longest CDS of the read is translated (table 11,
or for MetaProdigal peptides the table of the read's MetaProdigal model; trailing stop removed; an internal stop - DeepCDS S can predict through one - is kept as X and the query is reported
in a separate group <group>_internal_stop). Control: every query shuffled (same length and composition, seed 42), searched the same way.
All hits up to E = 10 are kept (best hit per query); hit rates are reported at several E-value thresholds.
Outputs: WORK/diamond/{queries,hits}*, RES/diamond_hit_rates.tsv
Usage: python 10_diamond_check.py --deepcds_gff <S gff> --db /dev/shm/nrt204_uniref90/uniref90 [--n_reads 2000000]
"""
import argparse
import glob
import gzip
import os
import random
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "/tmp/nrt204/FragmentPredictor/real_metagenome_poc/results"
DIAMOND = os.path.expanduser("~/.conda/envs/poc_env/bin/diamond")
COMP = str.maketrans("ACGTN", "TGCAN")
BASES = "TCAG"
CODE11 = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
TABLE = {a + b + c: CODE11[16 * i + 4 * j + k] for i, a in enumerate(BASES) for j, b in enumerate(BASES)
         for k, c in enumerate(BASES)}
THRESHOLDS = [1e-3, 1e-5, 1e-1]  # 1e-3 = DIAMOND's default E-value cut-off (main threshold)
FMT = ["qseqid", "sseqid", "pident", "length", "qlen", "slen", "evalue", "bitscore", "qframe"]


def rc(s):
    return s.translate(COMP)[::-1]


TABLE4 = {**TABLE, "TGA": "W"}  # MetaProdigal's meta mode uses translation-table-4 models for some reads


def translate(nt, table=11):
    t = TABLE4 if table == 4 else TABLE
    aa = "".join(t.get(nt[i:i + 3], "X") for i in range(0, len(nt) - 2, 3))
    return aa[:-1] if aa.endswith("*") else aa


def read_gff(paths, wanted, deepcds=False, tables=None):
    """{read: [(length, start, end, strand)]} for all CDS; DeepCDS _rc seqids -> strand 'rc'.
    tables (dict, Prodigal GFF only): filled with the translation table of each read's model."""
    out = {}
    for path in paths:
        with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as fh:
            seqhdr = None
            for line in fh:
                if line[0] == "#":
                    if tables is not None:
                        if line.startswith("# Sequence Data:"):
                            seqhdr = line.split('seqhdr="', 1)[1].split('"', 1)[0]
                        elif line.startswith("# Model Data:") and seqhdr in wanted:
                            tables[seqhdr] = int(line.split("transl_table=", 1)[1].split(";", 1)[0])
                    continue
                c = line.split("\t")
                if len(c) < 9 or c[2] != "CDS":
                    continue
                read, strand = c[0], c[6]
                if deepcds and read.endswith("_rc"):
                    read, strand = read[:-3], "rc"
                if read in wanted:
                    s, e = int(c[3]), int(c[4])
                    out.setdefault(read, []).append((e - s + 1, s, e, strand))
    return out


def cds_nt(seq, s, e, strand):
    if strand == "rc":
        return rc(seq)[s - 1:e]
    return rc(seq[s - 1:e]) if strand == "-" else seq[s - 1:e]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/real_metagenome_poc_work")
    parser.add_argument("--deepcds_gff", required=True)
    parser.add_argument("--db", default="/dev/shm/nrt204_uniref90/uniref90")
    parser.add_argument("--n_reads", type=int, default=2_000_000)
    parser.add_argument("--per_group", type=int, default=20_000)
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--out", default=None, help="default: WORK/diamond")
    parser.add_argument("--results", default=RES)
    args = parser.parse_args()
    out = args.out or f"{args.work}/diamond"
    os.makedirs(out, exist_ok=True)
    rnd = random.Random(42)

    seqs = {}
    proc = subprocess.Popen(["pigz", "-dc", f"{args.work}/fasta/reads_ge61.fa.gz"], stdout=subprocess.PIPE, text=True)
    name = None
    for line in proc.stdout:
        if line[0] == ">":
            if len(seqs) >= args.n_reads:
                break
            name = line[1:].rstrip()
        else:
            seqs[name] = line.rstrip()
    proc.kill()
    ids = list(seqs)
    wanted = set(ids)
    print(f"{len(ids):,} reads", flush=True)

    deep = read_gff([args.deepcds_gff], wanted, deepcds=True)
    last = ids[-1]
    assert last in deep or any(r in deep for r in ids[-1000:]), "DeepCDS GFF does not cover the read range"
    prod_table = {}
    prod = read_gff(sorted(glob.glob(f"{args.work}/preds/prodigal/chunk_*.gff.gz")), wanted, tables=prod_table)
    print(f"MetaProdigal translation tables: {pd.Series(prod_table).value_counts().to_dict()}", flush=True)
    fgs = read_gff(sorted(glob.glob(f"{args.work}/preds/fgs_complete/chunk_*.gff.gz")), wanted)
    cat = pd.read_csv(f"{args.work}/mapping/read_categories.tsv.gz", sep="\t", usecols=["read_id", "category"],
                      nrows=len(ids))
    assert list(cat["read_id"]) == ids
    category = dict(zip(cat["read_id"], cat["category"]))

    def longest(d, r, min_len=61):
        c = [x for x in d.get(r, []) if x[0] >= min_len]
        return max(c) if c else None

    CELLS = {(1, 1, 1): "all_tools", (1, 1, 0): "deepcds_prodigal", (1, 0, 1): "deepcds_fgs", (1, 0, 0): "deepcds_only",
             (0, 1, 1): "prodigal_fgs", (0, 1, 0): "prodigal_only", (0, 0, 1): "fgs_only"}
    PEPTIDE_SOURCE = {g: "deepcds" for g in ("all_tools", "deepcds_prodigal", "deepcds_fgs", "deepcds_only", "deepcds_short")}
    PEPTIDE_SOURCE.update(prodigal_fgs="prodigal", prodigal_only="prodigal", fgs_only="fgs")
    groups = {g: [] for g in (*CELLS.values(), "deepcds_short", "no_tool")}
    for r in ids:
        key = (bool(longest(deep, r)), bool(longest(prod, r)), bool(longest(fgs, r)))
        if any(key):
            groups[CELLS[tuple(int(k) for k in key)]].append(r)
        d = longest(deep, r)
        if r in deep and not d:
            groups["deepcds_short"].append(r)
        if r not in deep and r not in prod and r not in fgs:
            groups["no_tool"].append(r)
    print({g: len(v) for g, v in groups.items()}, flush=True)

    rows, internal_stops = [], {}
    pep, pep_shuf, nt, nt_shuf = [], [], [], []
    for g, members in groups.items():
        sample = rnd.sample(members, min(args.per_group, len(members)))
        internal_stops[g] = 0
        for r in sample:
            seq = seqs[r]
            if g == "no_tool":
                q = f"{g}|{category[r]}|{r}"
                nt.append(f">{q}\n{seq}\n")
                s = list(seq); rnd.shuffle(s)
                nt_shuf.append(f">{q}\n{''.join(s)}\n")
                rows.append((q, g, category[r], len(seq)))
                continue
            src = {"deepcds": deep, "prodigal": prod, "fgs": fgs}[PEPTIDE_SOURCE[g]]
            x = max(src[r]) if g == "deepcds_short" else longest(src, r)
            aa = translate(cds_nt(seq, x[1], x[2], x[3]), prod_table[r] if PEPTIDE_SOURCE[g] == "prodigal" else 11)
            g_q = g
            if "*" in aa:
                # DeepCDS S can predict through an in-frame stop (treated as a possible substitution error);
                # kept as X so a homologous protein can still be found, and reported as a separate group
                internal_stops[g] += 1
                aa, g_q = aa.replace("*", "X"), f"{g}_internal_stop"
            q = f"{g_q}|{category[r]}|{r}"
            pep.append(f">{q}\n{aa}\n")
            s = list(aa); rnd.shuffle(s)
            pep_shuf.append(f">{q}\n{''.join(s)}\n")
            rows.append((q, g_q, category[r], len(aa)))
    print("queries with an internal stop codon (kept as X, group <group>_internal_stop):", internal_stops, flush=True)
    for fname, lines in [("queries.faa", pep), ("queries_shuffled.faa", pep_shuf),
                         ("queries_nt.fa", nt), ("queries_nt_shuffled.fa", nt_shuf)]:
        with open(f"{out}/{fname}", "w") as fh:
            fh.write("".join(lines))

    common = ["-d", args.db, "--threads", str(args.threads), "--evalue", "10", "--max-target-seqs", "1",
              "--sensitive", "--outfmt", "6", *FMT, "--quiet"]
    for mode, q, o in [("blastp", "queries.faa", "hits.tsv"), ("blastp", "queries_shuffled.faa", "hits_shuffled.tsv"),
                       ("blastx", "queries_nt.fa", "hits_nt.tsv"), ("blastx", "queries_nt_shuffled.fa", "hits_nt_shuffled.tsv")]:
        print(f"diamond {mode} {q}", flush=True)
        subprocess.run([DIAMOND, mode, "-q", f"{out}/{q}", "-o", f"{out}/{o}", *common], check=True)

    def best(path):
        h = pd.read_csv(path, sep="\t", names=FMT)
        return h.sort_values("evalue").drop_duplicates("qseqid").set_index("qseqid")["evalue"]

    q = pd.DataFrame(rows, columns=["qseqid", "group", "category", "query_len"]).set_index("qseqid")
    q["evalue"] = pd.concat([best(f"{out}/hits.tsv"), best(f"{out}/hits_nt.tsv")])
    q["evalue_shuffled"] = pd.concat([best(f"{out}/hits_shuffled.tsv"), best(f"{out}/hits_nt_shuffled.tsv")])
    q.to_csv(f"{out}/per_query.tsv.gz", sep="\t", compression="gzip")
    summary = []
    for (g, c), sub in pd.concat([q.assign(category="all"), q]).groupby(["group", "category"], sort=False):
        row = {"group": g, "category": c, "n": len(sub), "median_len": sub["query_len"].median()}
        for t in THRESHOLDS:
            row[f"hit_E{t:g}"] = (sub["evalue"] <= t).mean()
            row[f"shuffled_E{t:g}"] = (sub["evalue_shuffled"] <= t).mean()
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(f"{args.results}/diamond_hit_rates.tsv", sep="\t", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
