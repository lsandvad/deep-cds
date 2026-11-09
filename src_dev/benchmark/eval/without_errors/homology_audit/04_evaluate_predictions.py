"""Step 4: TP/FN for every labeled test fragment, per tool, with the manuscript's CDS-level matching (exact match, IoU = 1).

Tools: DeepCDS N, MetaProdigal, FGS (Complete) on the error-free 300 bp test set. A labeled fragment is a TP if a
prediction on the same read is matched to it by benchmark_functions.find_matching_genes. Per-genome counts are
checked against calculate_metrics().

Outputs ($RESULTS/): per_fragment.tsv.gz (bins from step 3 + one boolean column {tool}_tp per tool)
                     per_genome_counts.tsv (genome x tool: tp, fp, fn)
Usage: python 04_evaluate_predictions.py [--work ...] [--results ...] [--procs 64]
"""
import argparse
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd

import benchmark_functions as bf

TOOLS = ["deepcds_n", "metaprodigal", "fgs_complete"]


def evaluate_genome(args):
    acc, staged = args
    with open(f"{staged}/testset/{acc}/testset_dict.pkl", "rb") as f:
        testset = bf.get_full_cds(pickle.load(f))
    with open(f"{staged}/testset/{acc}/read_names_list.pkl", "rb") as f:
        read_names = pickle.load(f)
    keys = [(r, i) for r, e in testset.items() for i in range(len(e["cds_coords"]))]
    pos = {k: n for n, k in enumerate(keys)}

    cols, counts = {}, []
    for tool in TOOLS:
        with open(f"{staged}/preds/{tool}/{acc}.pkl", "rb") as f:
            preds = bf.get_full_cds(pickle.load(f))
        tp = np.zeros(len(keys), dtype=bool)
        for read in read_names:
            actual = testset.get(read, {}).get("cds_coords", [])
            if actual:
                pairs, _ = bf.find_matching_genes(preds.get(read, {}).get("cds_coords", []), actual)
                for _, a in pairs:
                    tp[pos[(read, a)]] = True
        m = bf.calculate_metrics(testset, preds, read_names)
        assert m["total_true_positives"] == tp.sum(), (acc, tool)
        cols[f"{tool}_tp"] = tp
        counts.append({"genome": acc, "tool": tool, "tp": m["total_true_positives"],
                       "fp": m["total_false_positives"], "fn": m["total_false_negatives"]})
    df = pd.DataFrame({"genome": acc, "read_name": [k[0] for k in keys], "coord_idx": [k[1] for k in keys], **cols})
    return df, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/homology_audit_work")
    parser.add_argument("--results", default="/tmp/nrt204/FragmentPredictor/data/processed_data/homology_audit")
    parser.add_argument("--procs", type=int, default=64)
    args = parser.parse_args()
    os.makedirs(args.results, exist_ok=True)

    bins = pd.read_csv(f"{args.work}/bins.tsv.gz", sep="\t")
    accs = sorted(bins["genome"].unique())
    with Pool(args.procs) as pool:
        res = pool.map(evaluate_genome, [(a, f"{args.work}/staged") for a in accs], chunksize=1)

    tp = pd.concat([df for df, _ in res], ignore_index=True)
    frag = bins.merge(tp, on=["genome", "read_name", "coord_idx"], how="inner", validate="1:1")
    assert len(frag) == len(bins) == len(tp)
    frag.to_csv(f"{args.results}/per_fragment.tsv.gz", sep="\t", index=False)
    pd.DataFrame([c for _, cs in res for c in cs]).to_csv(f"{args.results}/per_genome_counts.tsv", sep="\t", index=False)
    print(f"{len(frag):,} fragments, {len(accs)} genomes")
    print("STEP 4 DONE")


if __name__ == "__main__":
    main()
