"""Step 8: FGS CDS with the last complete codon of the read restored, searched against UniRef90.

FGS never lets a CDS run to the last base of a read: its end is 1-3 bases short of the read end (0 bases in < 0.1%).
With 1-2 bases left these are a partial codon, but with 3 left (the reading frame that starts at the read's first
base, RF0) FGS drops a complete codon that MetaProdigal and DeepCDS keep; FGS therefore never predicts a 30 aa CDS
on a 90 bp read. Here, for all three FGS models, a CDS ending exactly 3 bases before the read end is extended by
those 3 bases, unless
  * it is on the + strand and already ends with a stop codon (a real 3' end), or
  * the added codon is a stop codon (on the CDS strand), or
  * FGS placed a deletion right after the reported end (Illumina modes; its N already takes the codon's place).
FGS (Complete): CDS from the step-4 GFF as in step 6. FGS (Illumina 5 / 10): indel-corrected CDS as in step 7
(.out files of step 7), extended at the read end before reverse-complementing - strand CDS.
Every CDS that is not extended must match step 6 (Complete: read, length) or step 7 (Illumina: read, peptide), row
by row; it keeps its peptide id and DIAMOND hit. Only the extended peptides are searched (ids f<chunk>_<n>), with the
same DIAMOND command as step 7. Control: shuffled residues of 157 random CDS per FGS model and chunk (seed 42 + chunk).
The FGS rows of RES/diamond_full_cds_{per_tool,per_cell}.tsv are replaced; the step-7 versions are kept as
RES/diamond_full_cds_{per_tool,per_cell}_fgs_no_rf0_codon.tsv.
Phases (each skipped if its output exists): rebuild + check + translate per chunk -> DIAMOND -> summary
Outputs: OUT/{chunks,hits.tsv,hits_control.tsv} (default WORK/diamond_fgs_rf0)
Usage: python 08_fgs_rf0_last_codon.py [--threads 192]
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
s07 = importlib.import_module("07_fgs_illumina_indel_corrected")
RES = "/tmp/nrt204/FragmentPredictor/real_metagenome_poc/results"
MODELS = {"FGS (Complete)": "complete", "FGS (Illumina 5)": "illumina_5", "FGS (Illumina 10)": "illumina_10"}
STOPS = {"TAA", "TAG", "TGA"}
N_CHUNKS = 128


def read_out_gz(path):
    """s07.read_out for a gzipped .out file."""
    tmp = gzip.open(path, "rt").read()
    indels, read = {}, None
    for line in tmp.splitlines():
        if line[0] == ">":
            read = line[1:]
        else:
            c = line.split("\t")
            indels[(read, int(c[0]), int(c[1]), c[2])] = ({int(p) for p in c[5][2:].split(",") if p},
                                                          {int(p) for p in c[6][2:].split(",") if p})
    return indels


def extend(read, e, strand, fwd, dels):
    """fwd (read orientation) plus the read's last 3 bases if FGS dropped a complete codon there, else fwd."""
    if len(read) - e != 3 or e + 1 in dels:
        return fwd, False
    add = read[e:e + 3]
    if (strand == "+" and (fwd[-3:] in STOPS or add in STOPS)) or (strand == "-" and dfn.rc(add) in STOPS):
        return fwd, False
    return fwd + add, True


def process_chunk(task):
    i, work, out, n_ctrl = task
    seqs = s07.read_fasta(f"{work}/chunks/chunk_{i:03d}.fa")
    old12 = pd.read_csv(f"{work}/diamond_full/chunks/cds_{i:03d}.tsv.gz", sep="\t")
    old16 = pd.read_csv(f"{work}/diamond_fgs_indel/chunks/cds_{i:03d}.tsv.gz", sep="\t")
    pep16 = s07.read_fasta(f"{work}/diamond_fgs_indel/chunks/peptides_{i:03d}.faa")
    cell = dict(old12[["read_id", "cell"]].drop_duplicates("read_id").itertuples(index=False))
    rows, pep_id, aa_of_row = [], {}, []
    n_ext = {}
    for tool, mode in MODELS.items():
        old = (old12 if mode == "complete" else old16).query("tool == @tool").reset_index(drop=True)
        indels = None if mode == "complete" else read_out_gz(f"{work}/diamond_fgs_indel/fgs/{mode}/chunk_{i:03d}.out.gz")
        cds = dfn.tool_gff_cds([f"{work}/preds/fgs_{mode}/chunk_{i:03d}.gff.gz"], seqs)
        if len(cds) != len(old):
            raise RuntimeError(f"chunk {i:03d} {tool}: {len(cds)} CDS, previous step had {len(old)}")
        n_ext[tool] = 0
        for k, (read, segs, strand) in enumerate(cds):
            s, e = segs[0]
            r = seqs[read]
            ins, dels = (set(), set()) if indels is None else indels[(read, s, e, strand)]
            fwd = r[s - 1:e] if indels is None else s07.corrected_cds(r, s, e, "+", ins, dels)
            nt, ext = extend(r, e, strand, fwd, dels)
            nt = dfn.rc(nt) if strand == "-" else nt
            aa = dfn.translate(nt, dfn.TABLE11)
            stop = "*" in aa
            aa = aa.replace("*", "X")
            o = old.iloc[k]
            if o["read_id"] != read:
                raise RuntimeError(f"chunk {i:03d} {tool}: row {k} is {read}, previous step has {o['read_id']}")
            if ext:
                n_ext[tool] += 1
                pid = pep_id.setdefault(aa, f"f{i:03d}_{len(pep_id)}")
                if len(aa) != o["aa_len"] + 1:
                    raise RuntimeError(f"chunk {i:03d} {tool}: {read} extended to {len(aa)} aa from {o['aa_len']}")
            else:
                pid = o["peptide_id"]
                same = len(aa) == o["aa_len"] if mode == "complete" else pep16[pid] == aa
                if not same:
                    raise RuntimeError(f"chunk {i:03d} {tool}: {read} {s}-{e} {strand} differs from the previous step")
            rows.append((tool, read, cell[read], bool(ins or dels), stop, len(aa), pid, ext))
            aa_of_row.append(aa)
    df = pd.DataFrame(rows, columns=["tool", "read_id", "cell", "indel_split", "in_frame_stop", "aa_len",
                                     "peptide_id", "extended"])
    with open(f"{out}/chunks/peptides_{i:03d}.faa", "w") as fh:
        fh.write("".join(f">{pid}\n{aa}\n" for aa, pid in pep_id.items()))
    rnd, ctrl = random.Random(42 + i), []
    for tool in MODELS:
        sub = df[df["tool"] == tool]
        for k, j in enumerate(sub.sample(min(n_ctrl, len(sub)), random_state=42 + i).index):
            s = list(aa_of_row[j]); rnd.shuffle(s)
            ctrl.append(f">ctrl|{tool}|{i}_{k}\n{''.join(s)}\n")
    with open(f"{out}/chunks/control_{i:03d}.faa", "w") as fh:
        fh.write("".join(ctrl))
    df.to_csv(f"{out}/chunks/cds_{i:03d}.tsv.gz.part", sep="\t", index=False, compression="gzip")
    os.rename(f"{out}/chunks/cds_{i:03d}.tsv.gz.part", f"{out}/chunks/cds_{i:03d}.tsv.gz")
    return i, len(df), n_ext, len(pep_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/real_metagenome_poc_work")
    parser.add_argument("--out", default="/tmp/nrt204/real_metagenome_poc_work/diamond_fgs_rf0")
    parser.add_argument("--db", default="/dev/shm/nrt204_uniref90/uniref90")
    parser.add_argument("--tmpdir", default="/dev/shm/nrt204_diamond_tmp", help="DIAMOND temporary files (large)")
    parser.add_argument("--n_control", type=int, default=20_000)
    parser.add_argument("--threads", type=int, default=192)
    parser.add_argument("--procs", type=int, default=64)
    parser.add_argument("--chunks", type=int, nargs="*", help="only these chunks, phase 1 only (test)")
    args = parser.parse_args()
    out = args.out
    os.makedirs(f"{out}/chunks", exist_ok=True)
    os.makedirs(args.tmpdir, exist_ok=True)
    n_ctrl = math.ceil(args.n_control / N_CHUNKS)

    # phase 1: rebuild + check + translate per chunk
    chunks = args.chunks if args.chunks else range(N_CHUNKS)
    todo = [(i, args.work, out, n_ctrl) for i in chunks if not os.path.exists(f"{out}/chunks/cds_{i:03d}.tsv.gz")]
    if todo:
        with Pool(min(args.procs, len(todo))) as pool:
            for i, n_cds, n_ext, n_pep in pool.imap_unordered(process_chunk, todo):
                print(f"  chunk {i:03d}: {n_cds:,} CDS, extended {n_ext}, {n_pep:,} new unique peptides", flush=True)
    if args.chunks:
        return

    # phase 2: DIAMOND (as step 7)
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

    # phase 3: summary (as step 7), FGS rows replaced in the results; hits of reused ids from steps 6 and 7
    s06.G["hit"] = set(pd.concat(pd.read_csv(f"{d}/hits.tsv", sep="\t", header=None, usecols=[0])[0]
                                 for d in (f"{args.work}/diamond_full", f"{args.work}/diamond_fgs_indel", out)))
    print(f"{len(s06.G['hit']):,} peptides with a hit (steps 6, 7, 8)", flush=True)
    tools = list(MODELS)
    ctrl_q = pd.Series([l[1:].split("|")[1] for l in open(f"{out}/control.faa") if l[0] == ">"]).value_counts()
    ctrl_hits = pd.read_csv(f"{out}/hits_control.tsv", sep="\t", names=fmt, usecols=["qseqid"])["qseqid"]
    ctrl_rate = ctrl_hits.drop_duplicates().str.split("|").str[1].value_counts().reindex(tools, fill_value=0) / ctrl_q
    print(f"control: {ctrl_q.sum():,} queries, {ctrl_hits.nunique():,} with a hit", flush=True)
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
        path, backup = f"{RES}/diamond_full_cds_{name}.tsv", f"{RES}/diamond_full_cds_{name}_fgs_no_rf0_codon.tsv"
        if not os.path.exists(backup):
            shutil.copy(path, backup)
        old = pd.read_csv(backup, sep="\t", index_col=idx)
        assert set(new.index) <= set(old.index), f"{name}: new rows not in the step-7 table"
        merged = old.copy()
        merged.loc[new.index, new.columns] = new
        merged.to_csv(path, sep="\t")
    pd.set_option("display.width", 220)
    print(per_tool.to_string(float_format=lambda x: f"{x:.3f}"))
    print(per_cell.to_string(float_format=lambda x: f"{x:.3f}"))
    print("STEP 18 DONE")


if __name__ == "__main__":
    mp.set_start_method("fork")  # workers inherit s06.G (Python 3.14 defaults to forkserver)
    main()
