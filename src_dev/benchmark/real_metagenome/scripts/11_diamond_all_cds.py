"""Step 6 (optional), exhaustive version of 10_diamond_check.py: every predicted CDS >= 61 bp is searched.

Population: the first --n_reads reads >= 61 bp (reads_ge61.fa.gz order). Tools: DeepCDS S, DeepCDS S+I,
MetaProdigal, FGS (Complete); every CDS >= 61 bp on either strand (no sampling).
  * DeepCDS: segments of one CDS split at a predicted indel (same group_id) are translated segment by segment and
    joined (indel-corrected protein); the length filter uses the full extent, as get_full_cds does. A CDS through
    an in-frame stop codon keeps the stop as X and is flagged (in_frame_stop).
  * MetaProdigal: translated with the genetic code of the MetaProdigal model chosen for the read (table 4 or 11).
  * Identical peptides are searched once and counted for every CDS / tool that predicts them.
DIAMOND blastp --sensitive against UniRef90, E <= 1e-3 (DIAMOND's default), best hit per query.
Control: 20,000 random CDS per tool, residues shuffled (seed 42), searched the same way.
Overlap cell of each CDS = which of DeepCDS S / MetaProdigal / FGS (Complete) detect its read (CDS >= 61 bp),
as in 10_diamond_check.py.
Outputs: WORK/diamond_all/{cds.tsv.gz, peptides.faa, hits.tsv, ...}, RES/diamond_all_cds_{per_tool,per_cell}.tsv
Usage: python 11_diamond_all_cds.py --gff_S <S gff> --gff_SI <S+I gff> [--n_reads 2000000]
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
TABLE11 = {a + b + c: CODE11[16 * i + 4 * j + k] for i, a in enumerate(BASES) for j, b in enumerate(BASES)
           for k, c in enumerate(BASES)}
TABLE4 = {**TABLE11, "TGA": "W"}
MIN_CDS = 61
TOOLS = ["DeepCDS S", "DeepCDS S+I", "MetaProdigal", "FGS (Complete)"]
CELL_NAMES = {(1, 1, 1): "all three", (1, 1, 0): "DeepCDS + MetaProdigal", (1, 0, 1): "DeepCDS + FGS",
              (1, 0, 0): "DeepCDS alone", (0, 1, 1): "MetaProdigal + FGS", (0, 1, 0): "MetaProdigal alone",
              (0, 0, 1): "FGS alone", (0, 0, 0): "none of S / MetaProdigal / FGS"}


def rc(s):
    return s.translate(COMP)[::-1]


def translate(nt, table):
    aa = "".join(table.get(nt[i:i + 3], "X") for i in range(0, len(nt) - 2, 3))
    return aa[:-1] if aa.endswith("*") else aa


def deepcds_cds(path, seqs):
    """[(read, [(start, end), ...] segments, oriented strand 'fwd'/'rc')] for DeepCDS CDS >= 61 bp (full extent)."""
    groups, out = {}, []
    with open(path) as fh:
        for line in fh:
            if line[0] == "#":
                continue
            c = line.split("\t")
            if len(c) < 9 or c[2] != "CDS":
                continue
            seqid = c[0]
            read, ori = (seqid[:-3], "rc") if seqid.endswith("_rc") else (seqid, "fwd")
            if read not in seqs:
                continue
            s, e = int(c[3]), int(c[4])
            gid = next((a[9:].rsplit(".", 1)[0] for a in c[8].split(";") if a.startswith("group_id=")), None)
            if gid is None:
                if e - s + 1 >= MIN_CDS:
                    out.append((read, [(s, e)], ori))
            else:
                groups.setdefault((seqid, gid), []).append((s, e))
    for (seqid, _), segs in groups.items():
        read, ori = (seqid[:-3], "rc") if seqid.endswith("_rc") else (seqid, "fwd")
        segs.sort()
        if segs[-1][1] - segs[0][0] + 1 >= MIN_CDS:
            out.append((read, segs, ori))
    return out


def tool_gff_cds(paths, seqs, tables=None):
    """[(read, [(start, end)], strand '+'/'-')] for Prodigal / FGS CDS >= 61 bp; tables: read -> genetic code."""
    out = []
    for path in paths:
        with gzip.open(path, "rt") as fh:
            seqhdr = None
            for line in fh:
                if line[0] == "#":
                    if tables is not None:
                        if line.startswith("# Sequence Data:"):
                            seqhdr = line.split('seqhdr="', 1)[1].split('"', 1)[0]
                        elif line.startswith("# Model Data:") and seqhdr in seqs:
                            tables[seqhdr] = int(line.split("transl_table=", 1)[1].split(";", 1)[0])
                    continue
                c = line.split("\t")
                if len(c) < 9 or c[2] != "CDS" or c[0] not in seqs:
                    continue
                s, e = int(c[3]), int(c[4])
                if e - s + 1 >= MIN_CDS:
                    out.append((c[0], [(s, e)], c[6]))
    return out


def peptide(seq, segs, strand, table):
    if strand == "rc":
        seq = rc(seq)
    if strand == "-":
        return translate(rc(seq[segs[0][0] - 1:segs[0][1]]), table)
    return "".join(translate(seq[s - 1:e], table) for s, e in segs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/real_metagenome_poc_work")
    parser.add_argument("--gff_S", required=True)
    parser.add_argument("--gff_SI", required=True)
    parser.add_argument("--db", default="/dev/shm/nrt204_uniref90/uniref90")
    parser.add_argument("--n_reads", type=int, default=2_000_000)
    parser.add_argument("--n_control", type=int, default=20_000)
    parser.add_argument("--threads", type=int, default=96)
    args = parser.parse_args()
    out = f"{args.work}/diamond_all"
    os.makedirs(out, exist_ok=True)

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
    print(f"{len(ids):,} reads; last {ids[-1]}", flush=True)

    tables = {}
    cds = {"DeepCDS S": deepcds_cds(args.gff_S, seqs), "DeepCDS S+I": deepcds_cds(args.gff_SI, seqs),
           "MetaProdigal": tool_gff_cds(sorted(glob.glob(f"{args.work}/preds/prodigal/chunk_*.gff.gz")), seqs, tables),
           "FGS (Complete)": tool_gff_cds(sorted(glob.glob(f"{args.work}/preds/fgs_complete/chunk_*.gff.gz")), seqs)}
    for t, path in (("DeepCDS S", args.gff_S), ("DeepCDS S+I", args.gff_SI)):
        # the DeepCDS GFF is written in input order (R1 reads first): it must extend past the last read in the range
        last = subprocess.run(f"tail -c 5000 '{path}' | tail -1 | cut -f1", shell=True, capture_output=True,
                              text=True).stdout.strip().removesuffix("_rc")
        spot, mate = (lambda r: (int(r.split(".")[1].rsplit("_", 1)[0]), int(r.rsplit("_", 1)[1])))(last)
        lspot, lmate = (lambda r: (int(r.split(".")[1].rsplit("_", 1)[0]), int(r.rsplit("_", 1)[1])))(ids[-1])
        assert (mate, spot) > (lmate, lspot), f"{t} GFF ends at {last}, before the last read {ids[-1]}"
    detected = {t: {r for r, _, _ in v} for t, v in cds.items()}
    print({t: f"{len(v):,} CDS on {len(detected[t]):,} reads" for t, v in cds.items()}, flush=True)

    cat = pd.read_csv(f"{args.work}/mapping/read_categories.tsv.gz", sep="\t", usecols=["read_id", "category"],
                      nrows=len(ids))
    assert list(cat["read_id"]) == ids
    category = dict(zip(cat["read_id"], cat["category"]))

    rows, pep_id = [], {}
    for tool, items in cds.items():
        for read, segs, strand in items:
            table = 4 if tool == "MetaProdigal" and tables.get(read) == 4 else 11
            aa = peptide(seqs[read], segs, strand, TABLE4 if table == 4 else TABLE11)
            stop = "*" in aa
            aa = aa.replace("*", "X")
            pid = pep_id.setdefault(aa, f"p{len(pep_id)}")
            key = tuple(int(read in detected[t]) for t in ("DeepCDS S", "MetaProdigal", "FGS (Complete)"))
            rows.append((tool, read, category[read], CELL_NAMES[key], len(segs) > 1, stop, len(aa), pid))
    df = pd.DataFrame(rows, columns=["tool", "read_id", "category", "cell", "indel_split", "in_frame_stop", "aa_len",
                                     "peptide_id"])
    with open(f"{out}/peptides.faa", "w") as fh:
        fh.write("".join(f">{pid}\n{aa}\n" for aa, pid in pep_id.items()))
    print(f"{len(df):,} CDS, {len(pep_id):,} unique peptides", flush=True)

    rnd = random.Random(42)
    ctrl = []
    for tool in TOOLS:
        sub = df[df["tool"] == tool].sample(min(args.n_control, (df["tool"] == tool).sum()), random_state=42)
        wanted = set(sub["peptide_id"])
        aa_of = {pid: aa for aa, pid in pep_id.items() if pid in wanted}
        for i, pid in enumerate(sub["peptide_id"]):
            s = list(aa_of[pid]); rnd.shuffle(s)
            ctrl.append(f">ctrl|{tool}|{i}\n{''.join(s)}\n")
    with open(f"{out}/control_shuffled.faa", "w") as fh:
        fh.write("".join(ctrl))

    fmt = ["qseqid", "sseqid", "pident", "length", "qlen", "slen", "evalue", "bitscore"]
    for q, o in [("peptides.faa", "hits.tsv"), ("control_shuffled.faa", "hits_control.tsv")]:
        if not os.path.exists(f"{out}/{o}"):
            print(f"diamond blastp {q}", flush=True)
            subprocess.run([DIAMOND, "blastp", "-q", f"{out}/{q}", "-d", args.db, "-o", f"{out}/{o}", "--threads",
                            str(args.threads), "--evalue", "1e-3", "--max-target-seqs", "1", "--sensitive",
                            "--outfmt", "6", *fmt, "--quiet"], check=True)
    hit = set(pd.read_csv(f"{out}/hits.tsv", sep="\t", names=fmt, usecols=["qseqid"])["qseqid"])
    df["hit"] = df["peptide_id"].isin(hit)
    ctrl_hits = pd.read_csv(f"{out}/hits_control.tsv", sep="\t", names=fmt, usecols=["qseqid"])["qseqid"]
    ctrl_rate = ctrl_hits.drop_duplicates().str.split("|").str[1].value_counts() / args.n_control
    df.to_csv(f"{out}/cds.tsv.gz", sep="\t", index=False, compression="gzip")

    def summarise(g):
        return pd.Series({"CDS": len(g), "with_hit": int(g["hit"].sum()), "hit_rate": g["hit"].mean(),
                          "CDS_in_frame_stop": int(g["in_frame_stop"].sum()),
                          "hit_rate_in_frame_stop": g.loc[g["in_frame_stop"], "hit"].mean(),
                          "reads": g["read_id"].nunique(), "median_aa": g["aa_len"].median()})

    per_tool = pd.concat([df.assign(category="all"), df]).groupby(["tool", "category"], sort=False).apply(summarise)
    per_tool["control_hit_rate"] = [ctrl_rate.get(t, 0.0) for t, _ in per_tool.index]
    per_cell = df.groupby(["cell", "tool"], sort=False).apply(summarise)
    per_tool.to_csv(f"{RES}/diamond_all_cds_per_tool.tsv", sep="\t")
    per_cell.to_csv(f"{RES}/diamond_all_cds_per_cell.tsv", sep="\t")
    pd.set_option("display.width", 220)
    print(per_tool.to_string(float_format=lambda x: f"{x:.3f}"))
    print(per_cell.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
