"""Step 6, full sample: every predicted CDS >= 61 bp on every read >= 61 bp is searched against UniRef90.

CDS extraction, translation and DIAMOND settings from diamond_functions.py.
  * all reads >= 61 bp (the 128 chunks of WORK/chunks, as used for MetaProdigal / FGS);
  * tools: DeepCDS N, S, S+I, MetaProdigal, FGS (Complete), FGS (Illumina 5), FGS (Illumina 10);
  * identical peptides are searched once per chunk (deduplicated within, not across, chunks);
  * control: ceil(--n_control / 128) random CDS per tool and chunk (seed 42 + chunk), residues shuffled.
Overlap cell of each CDS = which of DeepCDS S / MetaProdigal / FGS (Complete) detect its read.
DIAMOND blastp --sensitive against UniRef90, E <= 1e-3, best hit per query.
Phases (each skipped if its output exists): split DeepCDS GFFs by chunk -> translate per chunk -> DIAMOND -> summary
Outputs: OUT/{split,chunks,hits.tsv,...} (default /dev/shm), RES/diamond_full_cds_{per_tool,per_cell}.tsv
Usage: python 06_diamond_full_sample.py [--threads 192]
"""
import argparse
import glob
import gzip
import importlib
import math
import os
import random
import subprocess
import sys
import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
dfn = importlib.import_module("diamond_functions")
RES = "/tmp/nrt204/FragmentPredictor/real_metagenome_poc/results"
DEEPCDS = {"DeepCDS N": "model_without_errors", "DeepCDS S": "model_with_substitution_errors",
           "DeepCDS S+I": "model_with_errors"}
OTHER = {"MetaProdigal": "prodigal", "FGS (Complete)": "fgs_complete", "FGS (Illumina 5)": "fgs_illumina_5",
         "FGS (Illumina 10)": "fgs_illumina_10"}
TOOLS = list(DEEPCDS) + list(OTHER)
N_CHUNKS = 128
MAX_AA = 200
G = {}  # filled before forking: read -> chunk (phase 1), hit peptide ids (phase 4)


def split_gff(task):
    """Write the DeepCDS GFF lines of each chunk's reads (both orientations) to OUT/split/<model>/chunk_NNN.gff."""
    model, src, dst = task
    os.makedirs(dst, exist_ok=True)
    fhs = [open(f"{dst}/chunk_{i:03d}.gff.part", "w") for i in range(N_CHUNKS)]
    chunk_of = G["chunk_of"]
    with open(src) as fh:
        for line in fh:
            if line[0] == "#":
                continue
            seqid = line[:line.index("\t")]
            i = chunk_of.get(seqid[:-3] if seqid.endswith("_rc") else seqid)
            if i is not None:  # reads < 61 bp are not in any chunk
                fhs[i].write(line)
    for i, f in enumerate(fhs):
        f.close()
        os.rename(f"{dst}/chunk_{i:03d}.gff.part", f"{dst}/chunk_{i:03d}.gff")
    return model


def translate_chunk(task):
    i, work, out, n_ctrl = task
    seqs, name = {}, None
    with open(f"{work}/chunks/chunk_{i:03d}.fa") as fh:
        for line in fh:
            if line[0] == ">":
                name = line[1:].rstrip()
            else:
                seqs[name] = line.rstrip()
    tables = {}
    cds = {t: dfn.deepcds_cds(f"{out}/split/{m}/chunk_{i:03d}.gff", seqs) for t, m in DEEPCDS.items()}
    for t, d in OTHER.items():
        cds[t] = dfn.tool_gff_cds([f"{work}/preds/{d}/chunk_{i:03d}.gff.gz"], seqs,
                                  tables if t == "MetaProdigal" else None)
    detected = {t: {r for r, _, _ in cds[t]} for t in ("DeepCDS S", "MetaProdigal", "FGS (Complete)")}
    rows, pep_id = [], {}
    for tool, items in cds.items():
        for read, segs, strand in items:
            table = 4 if tool == "MetaProdigal" and tables.get(read) == 4 else 11
            aa = dfn.peptide(seqs[read], segs, strand, dfn.TABLE4 if table == 4 else dfn.TABLE11)
            stop = "*" in aa
            aa = aa.replace("*", "X")
            pid = pep_id.setdefault(aa, f"c{i:03d}_{len(pep_id)}")
            key = tuple(int(read in detected[t]) for t in ("DeepCDS S", "MetaProdigal", "FGS (Complete)"))
            rows.append((tool, read, dfn.CELL_NAMES[key], len(segs) > 1, stop, len(aa), pid))
    df = pd.DataFrame(rows, columns=["tool", "read_id", "cell", "indel_split", "in_frame_stop", "aa_len",
                                     "peptide_id"])
    with open(f"{out}/chunks/peptides_{i:03d}.faa", "w") as fh:
        fh.write("".join(f">{pid}\n{aa}\n" for aa, pid in pep_id.items()))
    aa_of = {pid: aa for aa, pid in pep_id.items()}
    rnd, ctrl = random.Random(42 + i), []
    for tool in TOOLS:
        sub = df[df["tool"] == tool]
        sub = sub.sample(min(n_ctrl, len(sub)), random_state=42 + i)
        for k, pid in enumerate(sub["peptide_id"]):
            s = list(aa_of[pid]); rnd.shuffle(s)
            ctrl.append(f">ctrl|{tool}|{i}_{k}\n{''.join(s)}\n")
    with open(f"{out}/chunks/control_{i:03d}.faa", "w") as fh:
        fh.write("".join(ctrl))
    df.to_csv(f"{out}/chunks/cds_{i:03d}.tsv.gz.part", sep="\t", index=False, compression="gzip")
    os.rename(f"{out}/chunks/cds_{i:03d}.tsv.gz.part", f"{out}/chunks/cds_{i:03d}.tsv.gz")
    return i, len(df), len(pep_id), len(seqs)


def summarise_chunk(path):
    """Per (tool, cell, in_frame_stop): CDS, CDS with hit, aa-length histogram; reads per tool and (tool, cell)."""
    df = pd.read_csv(path, sep="\t")
    df["hit"] = df["peptide_id"].isin(G["hit"])
    rows = []
    for (tool, cell, stop), g in df.groupby(["tool", "cell", "in_frame_stop"]):
        hist = np.bincount(np.minimum(g["aa_len"].to_numpy(), MAX_AA), minlength=MAX_AA + 1)
        rows.append((tool, cell, stop, len(g), int(g["hit"].sum()), hist))
    reads = {t: g["read_id"].nunique() for t, g in df.groupby("tool")}
    cell_reads = {k: g["read_id"].nunique() for k, g in df.groupby(["tool", "cell"])}
    return rows, reads, cell_reads


def median_from_hist(h):
    c = np.cumsum(h)
    return float(np.searchsorted(c, (c[-1] + 1) / 2)) if c[-1] else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/real_metagenome_poc_work")
    parser.add_argument("--out", default="/dev/shm/nrt204_poc_full")
    parser.add_argument("--db", default="/dev/shm/nrt204_uniref90/uniref90")
    parser.add_argument("--n_control", type=int, default=20_000)
    parser.add_argument("--threads", type=int, default=192)
    parser.add_argument("--procs", type=int, default=64)
    args = parser.parse_args()
    out = args.out
    os.makedirs(f"{out}/chunks", exist_ok=True)
    n_ctrl = math.ceil(args.n_control / N_CHUNKS)

    # phase 1: split the DeepCDS GFFs by chunk
    todo = [(t, f"{out}/deepcds_gff/{m}.gff", f"{out}/split/{m}") for t, m in DEEPCDS.items()
            if not os.path.exists(f"{out}/split/{m}/chunk_{N_CHUNKS - 1:03d}.gff")]
    if todo:
        chunk_of = {}
        for i in range(N_CHUNKS):
            with open(f"{args.work}/chunks/chunk_{i:03d}.fa") as fh:
                for line in fh:
                    if line[0] == ">":
                        chunk_of[line[1:].rstrip()] = i
        G["chunk_of"] = chunk_of
        print(f"{len(chunk_of):,} reads >= 61 bp in {N_CHUNKS} chunks; splitting {[t for t, _, _ in todo]}",
              flush=True)
        with Pool(len(todo)) as pool:
            for t in pool.imap_unordered(split_gff, todo):
                print(f"  split {t}", flush=True)
        G.clear()

    # phase 2: CDS -> peptides per chunk
    todo = [(i, args.work, out, n_ctrl) for i in range(N_CHUNKS) if not os.path.exists(f"{out}/chunks/cds_{i:03d}.tsv.gz")]
    if todo:
        with Pool(args.procs) as pool:
            for i, n_cds, n_pep, n_reads in pool.imap_unordered(translate_chunk, todo):
                print(f"  chunk {i:03d}: {n_reads:,} reads, {n_cds:,} CDS, {n_pep:,} unique peptides", flush=True)

    # phase 3: DIAMOND
    fmt = ["qseqid", "sseqid", "pident", "length", "qlen", "slen", "evalue", "bitscore"]
    for q, o in [("peptides", "hits.tsv"), ("control", "hits_control.tsv")]:
        if not os.path.exists(f"{out}/{o}"):
            subprocess.run(f"cat {out}/chunks/{q}_*.faa > {out}/{q}.faa", shell=True, check=True)
            n = int(subprocess.run(f"grep -c '>' {out}/{q}.faa", shell=True, capture_output=True, text=True).stdout)
            print(f"diamond blastp {q}.faa: {n:,} queries", flush=True)
            subprocess.run([dfn.DIAMOND, "blastp", "-q", f"{out}/{q}.faa", "-d", args.db, "-o", f"{out}/{o}.part",
                            "--threads", str(args.threads), "--evalue", "1e-3", "--max-target-seqs", "1",
                            "--sensitive", "--outfmt", "6", *fmt, "--quiet"], check=True)
            os.rename(f"{out}/{o}.part", f"{out}/{o}")

    # phase 4: summary
    G["hit"] = set(pd.read_csv(f"{out}/hits.tsv", sep="\t", names=fmt, usecols=["qseqid"])["qseqid"])
    print(f"{len(G['hit']):,} peptides with a hit", flush=True)
    ctrl_q = pd.Series([l[1:].split("|")[1] for l in open(f"{out}/control.faa") if l[0] == ">"]).value_counts()
    ctrl_hits = pd.read_csv(f"{out}/hits_control.tsv", sep="\t", names=fmt, usecols=["qseqid"])["qseqid"]
    ctrl_rate = ctrl_hits.drop_duplicates().str.split("|").str[1].value_counts().reindex(TOOLS, fill_value=0) / ctrl_q
    with Pool(args.procs) as pool:
        res = pool.map(summarise_chunk, sorted(glob.glob(f"{out}/chunks/cds_*.tsv.gz")))
    agg = pd.DataFrame([r for rows, _, _ in res for r in rows],
                       columns=["tool", "cell", "in_frame_stop", "CDS", "with_hit", "hist"])
    reads = pd.DataFrame([d for _, d, _ in res]).sum()
    cell_reads = pd.Series([(k, v) for _, _, d in res for k, v in d.items()]).pipe(
        lambda s: pd.DataFrame(s.tolist(), columns=["key", "n"]).groupby("key")["n"].sum())

    def summarise(g):
        stop = g[g["in_frame_stop"]]
        return pd.Series({"CDS": g["CDS"].sum(), "with_hit": g["with_hit"].sum(),
                          "hit_rate": g["with_hit"].sum() / g["CDS"].sum(),
                          "CDS_in_frame_stop": stop["CDS"].sum(),
                          "hit_rate_in_frame_stop": stop["with_hit"].sum() / stop["CDS"].sum() if len(stop) else np.nan,
                          "median_aa": median_from_hist(np.sum(list(g["hist"]), axis=0))})

    per_tool = agg.groupby("tool").apply(summarise).reindex(TOOLS)
    per_tool["reads"] = reads.reindex(TOOLS)
    per_tool["control_hit_rate"] = ctrl_rate.reindex(TOOLS)
    per_cell = agg.groupby(["cell", "tool"]).apply(summarise)
    per_cell["reads"] = [cell_reads.get((t, c), 0) for c, t in per_cell.index]
    per_cell = per_cell.reindex([(c, t) for c in dfn.CELL_NAMES.values() for t in TOOLS]).dropna(how="all")
    per_tool.to_csv(f"{RES}/diamond_full_cds_per_tool.tsv", sep="\t")
    per_cell.to_csv(f"{RES}/diamond_full_cds_per_cell.tsv", sep="\t")
    pd.set_option("display.width", 220)
    print(per_tool.to_string(float_format=lambda x: f"{x:.3f}"))
    print(per_cell.to_string(float_format=lambda x: f"{x:.3f}"))
    print("STEP 12 DONE")


if __name__ == "__main__":
    mp.set_start_method("fork")  # workers inherit G (Python 3.14 defaults to forkserver)
    main()
