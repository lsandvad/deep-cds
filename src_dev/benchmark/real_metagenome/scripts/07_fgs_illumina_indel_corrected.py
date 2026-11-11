"""Step 7: FGS (Illumina 5 / Illumina 10) CDS with FGS's own indel corrections, searched against UniRef90.

In the Illumina modes FragGeneScanRs predicts insertions and deletions in the CDS, but
  * the GFF coordinates (used in step 6) do not include them, so step 6 translated the uncorrected read sequence;
  * the .faa / .ffn repeat the last codon of most CDS (98% on chunk 000), so they cannot be used as they are.
Here FGS is rerun keeping the .out file (indel positions) and each CDS is rebuilt from the read: the base at each
predicted insertion (I:p) is removed and an N is put before each predicted deletion (D:p); - strand CDS are then
reverse-complemented. Every rebuilt CDS is checked against FGS's .ffn (identical, or identical after removing the
repeated final codon); any mismatch stops the run. The coordinates are unchanged (the rerun GFF must be identical
to the step-4 GFF), so the >= 61 bp filter, detection and read groups are as before.
Translation (table 11), DIAMOND (same command as step 6), control (shuffled residues) and summary as in step 6.
The FGS (Illumina 5) / (Illumina 10) rows of RES/diamond_full_cds_{per_tool,per_cell}.tsv are replaced;
the step-6 versions are kept as RES/diamond_full_cds_{per_tool,per_cell}_fgs_uncorrected.tsv.
Phases (each skipped if its output exists): rerun FGS + rebuild + translate per chunk -> DIAMOND -> summary
Usage: python 07_fgs_illumina_indel_corrected.py [--threads 192]
"""
import argparse
import glob
import gzip
import importlib
import math
import os
import random
import shutil
import subprocess
import sys
import multiprocessing as mp
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
dfn = importlib.import_module("diamond_functions")
s06 = importlib.import_module("06_diamond_full_sample")
RES = "/tmp/nrt204/FragmentPredictor/real_metagenome_poc/results"
FGS = os.path.expanduser("~/.conda/envs/gene_prediction_env/bin/FragGeneScanRs")
MODES = {"FGS (Illumina 5)": "illumina_5", "FGS (Illumina 10)": "illumina_10"}
N_CHUNKS = 128


def read_fasta(path):
    seqs, name = {}, None
    with open(path) as fh:
        for line in fh:
            if line[0] == ">":
                name = line[1:].rstrip()
                seqs[name] = []
            else:
                seqs[name].append(line.rstrip())
    return {k: "".join(v) for k, v in seqs.items()}


def read_out(path):
    """(read, start, end, strand) -> (insertion positions, deletion positions) from an FGS .out file."""
    indels, read = {}, None
    with open(path) as fh:
        for line in fh:
            if line[0] == ">":
                read = line[1:].rstrip()
            else:
                c = line.rstrip("\n").split("\t")
                ins = {int(p) for p in c[5][2:].split(",") if p}
                dels = {int(p) for p in c[6][2:].split(",") if p}
                indels[(read, int(c[0]), int(c[1]), c[2])] = (ins, dels)
    return indels


def corrected_cds(read, s, e, strand, ins, dels):
    """CDS as FGS corrects it: read[s..e] without inserted bases, N before deleted positions, oriented 5'->3'."""
    out = []
    for p in range(s, e + 1):
        if p in dels:
            out.append("N")
        if p not in ins:
            out.append(read[p - 1])
    if e + 1 in dels:  # FGS can place a deletion right after the reported end; its .ffn includes the N
        out.append("N")
    seq = "".join(out)
    return dfn.rc(seq) if strand == "-" else seq


def process_chunk(task):
    """Rerun FGS (both Illumina modes) on one chunk; rebuild, check and translate every CDS >= 61 bp."""
    i, work, out, n_ctrl = task
    seqs = read_fasta(f"{work}/chunks/chunk_{i:03d}.fa")
    cell = pd.read_csv(f"{out}/../diamond_full/chunks/cds_{i:03d}.tsv.gz", sep="\t", usecols=["read_id", "cell"])
    cell = dict(cell.drop_duplicates("read_id").itertuples(index=False))
    rows, pep_id = [], {}
    for tool, mode in MODES.items():
        d = f"{out}/fgs/{mode}"
        os.makedirs(d, exist_ok=True)
        prefix = f"{d}/chunk_{i:03d}"
        subprocess.run([FGS, "-s", f"{work}/chunks/chunk_{i:03d}.fa", "-t", mode, "-w", "0", "-o", prefix],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        same = subprocess.run(f"zcat {work}/preds/fgs_{mode}/chunk_{i:03d}.gff.gz | cmp -s - {prefix}.gff",
                              shell=True).returncode == 0
        if not same:
            raise RuntimeError(f"chunk {i:03d} {mode}: rerun GFF differs from the step-4 GFF")
        indels = read_out(f"{prefix}.out")
        ffn = read_fasta(f"{prefix}.ffn")
        for read, segs, strand in dfn.tool_gff_cds([f"{work}/preds/fgs_{mode}/chunk_{i:03d}.gff.gz"], seqs):
            s, e = segs[0]
            ins, dels = indels[(read, s, e, strand)]
            nt = corrected_cds(seqs[read], s, e, strand, ins, dels)
            f = ffn[f"{read}_{s}_{e}_{strand}"]
            if nt != f and nt != (f[:-3] if strand == "+" else f[3:]):  # repeated codon is 3' of the CDS
                raise RuntimeError(f"chunk {i:03d} {mode}: {read} {s}-{e} {strand} does not match the FGS .ffn")
            aa = dfn.translate(nt, dfn.TABLE11)
            stop = "*" in aa
            aa = aa.replace("*", "X")
            pid = pep_id.setdefault(aa, f"g{i:03d}_{len(pep_id)}")
            rows.append((tool, read, cell[read], bool(ins or dels), stop, len(aa), pid))
        subprocess.run(["gzip", "-f", f"{prefix}.out"], check=True)
        for ext in ("gff", "ffn", "faa"):
            os.remove(f"{prefix}.{ext}")
    df = pd.DataFrame(rows, columns=["tool", "read_id", "cell", "indel_split", "in_frame_stop", "aa_len",
                                     "peptide_id"])
    with open(f"{out}/chunks/peptides_{i:03d}.faa", "w") as fh:
        fh.write("".join(f">{pid}\n{aa}\n" for aa, pid in pep_id.items()))
    aa_of = {pid: aa for aa, pid in pep_id.items()}
    rnd, ctrl = random.Random(42 + i), []
    for tool in MODES:
        sub = df[df["tool"] == tool]
        sub = sub.sample(min(n_ctrl, len(sub)), random_state=42 + i)
        for k, pid in enumerate(sub["peptide_id"]):
            s = list(aa_of[pid]); rnd.shuffle(s)
            ctrl.append(f">ctrl|{tool}|{i}_{k}\n{''.join(s)}\n")
    with open(f"{out}/chunks/control_{i:03d}.faa", "w") as fh:
        fh.write("".join(ctrl))
    df.to_csv(f"{out}/chunks/cds_{i:03d}.tsv.gz.part", sep="\t", index=False, compression="gzip")
    os.rename(f"{out}/chunks/cds_{i:03d}.tsv.gz.part", f"{out}/chunks/cds_{i:03d}.tsv.gz")
    return i, len(df), int(df["indel_split"].sum()), len(pep_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/real_metagenome_poc_work")
    parser.add_argument("--out", default="/tmp/nrt204/real_metagenome_poc_work/diamond_fgs_indel")
    parser.add_argument("--db", default="/dev/shm/nrt204_uniref90/uniref90")
    parser.add_argument("--tmpdir", default="/dev/shm/nrt204_diamond_tmp", help="DIAMOND temporary files (large)")
    parser.add_argument("--n_control", type=int, default=20_000)
    parser.add_argument("--threads", type=int, default=192)
    parser.add_argument("--procs", type=int, default=128)
    args = parser.parse_args()
    out = args.out
    os.makedirs(f"{out}/chunks", exist_ok=True)
    os.makedirs(args.tmpdir, exist_ok=True)
    n_ctrl = math.ceil(args.n_control / N_CHUNKS)

    # phase 1: rerun FGS, rebuild + check + translate per chunk
    todo = [(i, args.work, out, n_ctrl) for i in range(N_CHUNKS) if not os.path.exists(f"{out}/chunks/cds_{i:03d}.tsv.gz")]
    if todo:
        with Pool(args.procs) as pool:
            for i, n_cds, n_indel, n_pep in pool.imap_unordered(process_chunk, todo):
                print(f"  chunk {i:03d}: {n_cds:,} CDS ({n_indel:,} with indels) all match the FGS .ffn, "
                      f"{n_pep:,} unique peptides", flush=True)

    # phase 2: DIAMOND (as step 6)
    fmt = ["qseqid", "sseqid", "pident", "length", "qlen", "slen", "evalue", "bitscore"]
    for q, o in [("peptides", "hits.tsv"), ("control", "hits_control.tsv")]:
        if not os.path.exists(f"{out}/{o}"):
            subprocess.run(f"cat {out}/chunks/{q}_*.faa > {out}/{q}.faa", shell=True, check=True)
            n = int(subprocess.run(f"grep -c '>' {out}/{q}.faa", shell=True, capture_output=True, text=True).stdout)
            print(f"diamond blastp {q}.faa: {n:,} queries", flush=True)
            subprocess.run([dfn.DIAMOND, "blastp", "-q", f"{out}/{q}.faa", "-d", args.db, "-o", f"{out}/{o}.part",
                            "--threads", str(args.threads), "--evalue", "1e-3", "--max-target-seqs", "1",
                            "--sensitive", "--tmpdir", args.tmpdir, "--outfmt", "6", *fmt, "--quiet"], check=True)
            os.rename(f"{out}/{o}.part", f"{out}/{o}")

    # phase 3: summary (as step 6), FGS Illumina rows replaced in the step-6 results
    s06.G["hit"] = set(pd.read_csv(f"{out}/hits.tsv", sep="\t", names=fmt, usecols=["qseqid"])["qseqid"])
    print(f"{len(s06.G['hit']):,} peptides with a hit", flush=True)
    tools = list(MODES)
    ctrl_q = pd.Series([l[1:].split("|")[1] for l in open(f"{out}/control.faa") if l[0] == ">"]).value_counts()
    ctrl_hits = pd.read_csv(f"{out}/hits_control.tsv", sep="\t", names=fmt, usecols=["qseqid"])["qseqid"]
    ctrl_rate = ctrl_hits.drop_duplicates().str.split("|").str[1].value_counts().reindex(tools, fill_value=0) / ctrl_q
    with Pool(args.procs) as pool:
        res = pool.map(s06.summarise_chunk, sorted(glob.glob(f"{out}/chunks/cds_*.tsv.gz")))
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
                          "hit_rate_in_frame_stop": stop["with_hit"].sum() / stop["CDS"].sum() if len(stop) else float("nan"),
                          "median_aa": s06.median_from_hist(sum(g["hist"]))})

    per_tool = agg.groupby("tool").apply(summarise).reindex(tools)
    per_tool["reads"] = reads.reindex(tools)
    per_tool["control_hit_rate"] = ctrl_rate.reindex(tools)
    per_cell = agg.groupby(["cell", "tool"]).apply(summarise)
    per_cell["reads"] = [cell_reads.get((t, c), 0) for c, t in per_cell.index]

    for name, new, idx in [("per_tool", per_tool, ["tool"]), ("per_cell", per_cell, ["cell", "tool"])]:
        path, backup = f"{RES}/diamond_full_cds_{name}.tsv", f"{RES}/diamond_full_cds_{name}_fgs_uncorrected.tsv"
        if not os.path.exists(backup):
            shutil.copy(path, backup)
        old = pd.read_csv(backup, sep="\t", index_col=idx)
        assert set(new.index) <= set(old.index), f"{name}: new rows not in the step-6 table"
        merged = old.copy()
        merged.loc[new.index, new.columns] = new
        merged.to_csv(path, sep="\t")
    pd.set_option("display.width", 220)
    print(per_tool.to_string(float_format=lambda x: f"{x:.3f}"))
    print(per_cell.to_string(float_format=lambda x: f"{x:.3f}"))
    print("STEP 16 DONE")


if __name__ == "__main__":
    mp.set_start_method("fork")  # workers inherit s06.G (Python 3.14 defaults to forkserver)
    main()
