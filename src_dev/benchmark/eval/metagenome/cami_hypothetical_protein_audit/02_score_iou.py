"""Re-score the CAMI CDS fragments at a lower IoU threshold, reusing the categories from 01_classify_and_score.py.

TP per fragment and model: find_matching_genes(..., overlap_criteria=IOU) of the CAMI benchmark
(../cami_main_benchmark/cami_benchmark_functions.py), exactly as 01 does at IoU = 1.
Output: $RESULTS/hypothetical_counts_iou<IOU>.tsv (same columns as hypothetical_counts.tsv)
Usage: python 02_score_iou.py --iou 0.8 [--work /tmp/nrt204/hypothetical_audit_work]
"""
import argparse
import importlib.util
import os
import pickle
import sys
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = "/tmp/nrt204/FragmentPredictor/data/processed_data/cami_hypothetical_protein_audit"
spec = importlib.util.spec_from_file_location("step1", os.path.join(HERE, "01_classify_and_score.py"))
step1 = importlib.util.module_from_spec(spec)
sys.modules["step1"] = step1  # so forked workers can resolve step1.match_model
spec.loader.exec_module(step1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iou", type=float, required=True)
    parser.add_argument("--work", default="/tmp/nrt204/hypothetical_audit_work")
    args = parser.parse_args()
    step1.OVERLAP = args.iou

    with open(f"{step1.D}/processed_data/testset_processed/{step1.VARIANT}/{step1.SAMPLE}/testset_dict.pkl", "rb") as f:
        step1.testset_dict = step1.get_full_cds(pickle.load(f))
    step1.keys = [(r, i) for r, e in step1.testset_dict.items() for i in range(len(e["cds_coords"]))]
    with Pool(7) as pool:
        tp_flags = dict(pool.map(step1.match_model, list(step1.PRED_PATHS)))

    df = pd.read_csv(f"{args.work}/fragments.tsv.gz", sep="\t",
                     usecols=["read_name", "coord_idx", "genome_bin_id", "domain", "category", "length_bp"])
    assert list(zip(df["read_name"], df["coord_idx"])) == step1.keys, "fragment order differs from step 01"
    for model, tp in tp_flags.items():
        df[model] = tp


    df["length_class"] = pd.cut(df["length_bp"], step1.LENGTH_BINS, labels=step1.LENGTH_LABELS).astype(str)
    long = df.melt(id_vars=["genome_bin_id", "domain", "category", "length_class"], value_vars=list(step1.PRED_PATHS),
                   var_name="model", value_name="tp")
    agg = (long.groupby(["model", "genome_bin_id", "domain", "category", "length_class"], observed=True)["tp"]
           .agg(n="size", tp="sum").reset_index())
    agg.to_csv(f"{RESULTS}/hypothetical_counts_iou{args.iou:g}.tsv", sep="\t", index=False)
    print("DONE")


if __name__ == "__main__":
    main()
