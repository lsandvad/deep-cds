"""Step 3: extract and translate the labeled CAMI CDS fragments, and record TP/FN per fragment and model.

Test set: CAMI_metagenome_no_ab_initio_no_cds_contigs, sample_0. A fragment = one entry of
testset_dict.pkl after get_full_cds() (as scored in step 1).
Translation: each labeled piece [start, end, rf] in its own frame on the read, complete codons only, NCBI
table 11; pieces of an indel-disrupted CDS ('disrupted_rf', original pieces kept by get_full_cds) are
translated separately and concatenated; terminal stop removed, internal stops -> X (counted).
TP per model: find_matching_genes(..., overlap_criteria=1) (cami_benchmark_functions.py),
on all predictions and labels of the read, exactly as in step 1. Totals are checked against step 1.

Outputs ($WORK/fragments/): query.fasta (fragments >= 20 aa), fragments.tsv.gz (fragment_id, read_name,
coord_idx, genome_bin_id, domain, n_pieces, aa_len, n_internal_stops, TP flag per model)
Usage: python 03_extract_cami_fragments.py [--work /tmp/nrt204/cami_overlap_work]
"""
import argparse
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from Bio.Seq import Seq

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = "/tmp/nrt204/FragmentPredictor/data/processed_data/cami_main_benchmark"
sys.path.insert(0, HERE)
from cami_benchmark_functions import find_matching_genes, get_full_cds  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("step1", os.path.join(HERE, "01_per_bin_counts.py"))
step1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step1)

project_root, eval_data_dir, sample = step1.project_root, step1.eval_data_dir, step1.sample
MODELS = step1.model_names
MIN_AA = 20

testset_dict = None
keys = None


def translate_piece(read, start, end, rf):
    nt = read[start - 1:end]
    return str(Seq(nt[: len(nt) // 3 * 3]).translate(table=11))


def match_model(model):
    with open(step1.PRED_PATHS[model], "rb") as f:
        preds = get_full_cds(pickle.load(f))
    tp = np.zeros(len(keys), dtype=bool)
    pos = {k: i for i, k in enumerate(keys)}
    for read_name, entry in testset_dict.items():
        actual = entry["cds_coords"]
        pairs, _ = find_matching_genes(preds.get(read_name, {}).get("cds_coords", []), actual, step1.overlap_criteria)
        for _, a in pairs:
            tp[pos[(read_name, a)]] = True
    print(f"  {model}: {tp.sum():,} TP", flush=True)
    return model, tp


def main():
    global testset_dict, keys
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/cami_overlap_work")
    args = parser.parse_args()
    out = f"{args.work}/fragments"
    os.makedirs(out, exist_ok=True)

    with open(f"{project_root}/data/processed_data/testset_processed/{eval_data_dir}/{sample}/testset_dict.pkl", "rb") as f:
        testset_dict = get_full_cds(pickle.load(f))
    keys = [(r, i) for r, e in testset_dict.items() for i in range(len(e["cds_coords"]))]
    print(f"{len(testset_dict):,} reads with labeled CDS, {len(keys):,} labeled CDS fragments", flush=True)

    with Pool(len(MODELS)) as pool:
        tp_flags = dict(pool.map(match_model, MODELS))

    meta = pd.read_csv(f"{project_root}/data/processed_data/reads_processed/test/CAMI_metagenome/{sample}_cds_labels.tsv.gz",
                       sep="\t", compression="gzip", usecols=["read_name", "read", "genome_bin_id", "domain"])
    meta = meta[meta["read_name"].isin(testset_dict.keys())].set_index("read_name")

    rows, n_short = [], 0
    with open(f"{out}/query.fasta", "w") as fasta:
        for n, (read_name, i) in enumerate(keys):
            coord = testset_dict[read_name]["cds_coords"][i]
            read = meta.at[read_name, "read"]
            pieces = coord[3] if coord[2] == "disrupted_rf" else [coord]
            aa = "".join(translate_piece(read, s, e, rf) for s, e, rf in pieces)
            aa = aa[:-1] if aa.endswith("*") else aa
            n_stops = aa.count("*")
            aa = aa.replace("*", "X")
            fid = f"c{n:09d}"
            if len(aa) >= MIN_AA:
                fasta.write(f">{fid}\n{aa}\n")
            else:
                n_short += 1
            rows.append((fid, read_name, i, meta.at[read_name, "genome_bin_id"], meta.at[read_name, "domain"],
                         len(pieces), len(aa), n_stops))
    df = pd.DataFrame(rows, columns=["fragment_id", "read_name", "coord_idx", "genome_bin_id", "domain",
                                     "n_pieces", "aa_len", "n_internal_stops"])
    for m in MODELS:
        df[m] = tp_flags[m]

    # Check against step 1: TP per model must equal the per-bin counts' TP totals
    counts = pd.read_csv(f"{RESULTS}/per_bin_counts.tsv", sep="\t")
    for m in MODELS:
        assert int(df[m].sum()) == int(counts.loc[counts["model"] == m, "tp"].sum()), m
        assert len(df) == int((counts.loc[counts["model"] == m, "tp"] + counts.loc[counts["model"] == m, "fn"]).sum()), m
    df.to_csv(f"{out}/fragments.tsv.gz", sep="\t", index=False, compression="gzip")
    print(f"fragments: {len(df):,}; < {MIN_AA} aa (not searched, binned as 'no hit'): {n_short:,}; "
          f"with internal stops: {(df['n_internal_stops'] > 0).sum():,}; indel-disrupted: {(df['n_pieces'] > 1).sum():,}")
    print("Check passed: TP and labeled-CDS totals equal step 1 for all models")
    print("STEP 3 DONE")


if __name__ == "__main__":
    main()
