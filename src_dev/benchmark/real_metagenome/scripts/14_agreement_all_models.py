"""Step 8: hit rates per read group, with the groups defined by all DeepCDS models (and all FGS models).

Same CDS and DIAMOND hits as 12_diamond_full_sample.py (its per-chunk CDS tables and hits.tsv), and for the three FGS
models the CDS of 18_fgs_rf0_last_codon.py (last codon restored; Illumina modes indel-corrected as in step 16), with
the hits of steps 12, 16 and 18. The read group of a CDS is which
tools predict a CDS >= 61 bp (>= 20 aa) on its read, for two tool sets:
  * all_models: DeepCDS N, S, S+I, MetaProdigal, FGS (Complete) (31 groups);
  * all_tools: the same plus FGS (Illumina 5) and FGS (Illumina 10) (127 groups).
Outputs: RES/diamond_full_agreement_{all_models,all_tools}.tsv (one row per group and tool), printed summary.
Usage: python 14_agreement_all_models.py [--procs 64]
"""
import argparse
import os
import multiprocessing as mp
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "/tmp/nrt204/FragmentPredictor/real_metagenome_poc/results"
MODELS = ["DeepCDS N", "DeepCDS S", "DeepCDS S+I", "MetaProdigal", "FGS (Complete)"]
TOOL_SETS = {"all_models": (MODELS, ["N", "S", "S+I", "MP", "FGS"]),
             "all_tools": (MODELS + ["FGS (Illumina 5)", "FGS (Illumina 10)"],
                           ["N", "S", "S+I", "MP", "FGS", "FGS-0.5", "FGS-1.0"])}
N_CHUNKS = 128
G = {}  # chunk -> set of peptide ids with a hit (steps 12, 16 and 18), filled before forking


def summarise_chunk(task):
    i, path12, path18 = task
    df = pd.read_csv(path12, sep="\t", usecols=["tool", "read_id", "aa_len", "peptide_id"])
    df = pd.concat([df[~df["tool"].str.startswith("FGS")],
                    pd.read_csv(path18, sep="\t", usecols=["tool", "read_id", "aa_len", "peptide_id"])])
    df["hit"] = df["peptide_id"].isin(G["hit"].get(i, set()))
    out = {}
    for name, (tools, _) in TOOL_SETS.items():
        d = df[df["tool"].isin(tools)].copy()
        d["bit"] = d["tool"].map({t: 1 << k for k, t in enumerate(tools)})
        read_mask = d.drop_duplicates(["read_id", "tool"]).groupby("read_id")["bit"].sum()
        d["group"] = d["read_id"].map(read_mask)
        cds = d.groupby(["group", "tool"]).agg(CDS=("hit", "size"), with_hit=("hit", "sum")).reset_index()
        reads = read_mask.value_counts().rename_axis("group").reset_index(name="reads")
        out[name] = (cds, reads)
    return out, int(df["aa_len"].min())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/real_metagenome_poc_work")
    parser.add_argument("--procs", type=int, default=64)
    args = parser.parse_args()

    hits = pd.concat(pd.read_csv(f"{args.work}/{d}/hits.tsv", sep="\t", header=None, usecols=[0], names=["q"])["q"]
                     for d in ("diamond_full", "diamond_fgs_indel", "diamond_fgs_rf0")).drop_duplicates()
    G["hit"] = {}
    for c, g in hits.groupby(hits.str.split("_", n=1).str[0]):  # c000_* (step 12), g000_* (step 16), f000_* (step 18)
        G["hit"].setdefault(int(c[1:]), set()).update(g)
    print(f"{len(hits):,} peptides with a hit", flush=True)

    tasks = [(i, f"{args.work}/diamond_full/chunks/cds_{i:03d}.tsv.gz",
              f"{args.work}/diamond_fgs_rf0/chunks/cds_{i:03d}.tsv.gz") for i in range(N_CHUNKS)]
    with Pool(args.procs) as pool:
        res = pool.map(summarise_chunk, tasks)
    print(f"min CDS length: {min(r[1] for r in res)} aa")

    for name, (tools, short) in TOOL_SETS.items():
        cds = pd.concat([r[0][name][0] for r in res]).groupby(["group", "tool"], as_index=False).sum()
        reads = pd.concat([r[0][name][1] for r in res]).groupby("group")["reads"].sum()
        cds["reads"] = cds["group"].map(reads)
        cds["hit_rate"] = cds["with_hit"] / cds["CDS"]
        cds["n_tools"] = cds["group"].map(lambda m: bin(m).count("1"))
        cds["group_label"] = cds["group"].map(lambda m: " + ".join(s for k, s in enumerate(short) if m >> k & 1))
        cds = cds.sort_values(["n_tools", "reads", "group"], ascending=[False, False, True])
        cds.to_csv(f"{RES}/diamond_full_agreement_{name}.tsv", sep="\t", index=False)

        wide = cds.pivot_table(index=["n_tools", "group_label", "reads"], columns="tool", values="hit_rate")[tools]
        wide = wide.sort_index(level=[0, 2], ascending=[False, False])
        print(f"\n{name}: reads with a CDS from >= 1 of the {len(tools)} tools: {reads.sum():,}; {len(wide)} groups")
        print(wide.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
