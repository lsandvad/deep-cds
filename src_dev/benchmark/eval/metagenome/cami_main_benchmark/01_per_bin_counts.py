"""Step 1: CDS-level TP/FP/FN per CAMI genome bin and model (CAMI_metagenome_no_ab_initio_no_cds_contigs).

CDS-level evaluation of the CAMI benchmark (matching functions in cami_benchmark_functions.py, exact match,
overlap_criteria = 1): every read's
TP/FP/FN is attributed to the genome bin the read came from (reads without metadata -> "Unknown").
Because counts are per bin, a stricter test set without some genomes is obtained by dropping their bins
and re-pooling - identical to re-running the evaluation on the reduced read set.

Output ($RESULTS): per_bin_counts.tsv (model, genome_bin_id, domain, tp, fp, fn); with --variant <other CAMI test-set
        variant> (e.g. CAMI_metagenome, CAMI_metagenome_no_ab_initio, CAMI_metagenome_no_cds_contigs):
        per_bin_counts_<variant>.tsv
Usage: python 01_per_bin_counts.py [--variant CAMI_metagenome_no_ab_initio_no_cds_contigs]
"""
import argparse
import os
import pickle
import sys
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = "/tmp/nrt204/FragmentPredictor/data/processed_data/cami_main_benchmark"
sys.path.insert(0, HERE)
from cami_benchmark_functions import find_matching_genes, get_full_cds  # noqa: E402

project_root = "/tmp/nrt204/FragmentPredictor"
data_dir = "CAMI_metagenome"
DEFAULT_VARIANT = "CAMI_metagenome_no_ab_initio_no_cds_contigs"
_parser = argparse.ArgumentParser()
_parser.add_argument("--variant", default=DEFAULT_VARIANT)
eval_data_dir = _parser.parse_args().variant
sample = "sample_0"
overlap_criteria = 1
model_names = ["fgs_complete", "fgs_illumina_5", "fgs_illumina_10", "prodigal",
               "deep_cds_no_errors", "deep_cds_substitution", "deep_cds_errors"]
CKPT = "full_model_all_genomes_seed_42_trained_final_8M_no_dropout"
PRED_PATHS = {
    "prodigal": f"{project_root}/data/processed_data/predictions/processed_predictions/prodigal_preds/{eval_data_dir}/{sample}/model_preds_dict.pkl",
    "fgs_complete": f"{project_root}/data/processed_data/predictions/processed_predictions/fgs_preds/{eval_data_dir}_complete/{sample}.pkl",
    "fgs_illumina_5": f"{project_root}/data/processed_data/predictions/processed_predictions/fgs_preds/{eval_data_dir}_illumina_5/{sample}.pkl",
    "fgs_illumina_10": f"{project_root}/data/processed_data/predictions/processed_predictions/fgs_preds/{eval_data_dir}_illumina_10/{sample}.pkl",
    "deep_cds_no_errors": f"{project_root}/data/processed_data/predictions/processed_predictions/DeepCDS/model_without_errors/{eval_data_dir}/{CKPT}/{sample}/model_preds_dict.pkl",
    "deep_cds_substitution": f"{project_root}/data/processed_data/predictions/processed_predictions/DeepCDS/model_with_substitution_errors/{eval_data_dir}/{CKPT}/{sample}/model_preds_dict.pkl",
    "deep_cds_errors": f"{project_root}/data/processed_data/predictions/processed_predictions/DeepCDS/model_with_errors/{eval_data_dir}/{CKPT}/{sample}/model_preds_dict.pkl",
}

# Loaded once in the parent; forked workers share them
testset_dict = None
read_meta = None


def count_model(model):
    with open(PRED_PATHS[model], "rb") as f:
        preds = get_full_cds(pickle.load(f))
    bin_counts = {}
    for read_name in testset_dict.keys() | preds.keys():
        actual = testset_dict.get(read_name, {}).get("cds_coords", [])
        predicted = preds.get(read_name, {}).get("cds_coords", [])
        _, tp = find_matching_genes(predicted, actual, overlap_criteria)
        bc = bin_counts.setdefault(read_meta.get(read_name, ("Unknown", "Unknown"))[0], {"tp": 0, "fp": 0, "fn": 0})
        bc["tp"] += tp
        bc["fp"] += len(predicted) - tp
        bc["fn"] += len(actual) - tp
    print(f"  {model}: done", flush=True)
    return model, bin_counts


def main():
    global testset_dict, read_meta
    with open(f"{project_root}/data/processed_data/testset_processed/{eval_data_dir}/{sample}/testset_dict.pkl", "rb") as f:
        testset_dict = get_full_cds(pickle.load(f))
    meta = pd.read_csv(f"{project_root}/data/processed_data/reads_processed/test/{data_dir}/{sample}_cds_labels.tsv.gz",
                       sep="\t", compression="gzip", usecols=["read_name", "genome_bin_id", "domain"])
    read_meta = dict(zip(meta["read_name"], zip(meta["genome_bin_id"], meta["domain"])))
    bin_domain = dict(zip(meta["genome_bin_id"], meta["domain"]))
    del meta
    print(f"testset reads with CDS: {len(testset_dict):,}; reads with metadata: {len(read_meta):,}", flush=True)

    with Pool(len(model_names)) as pool:
        results = dict(pool.map(count_model, model_names))

    rows = [{"model": m, "genome_bin_id": b, "domain": bin_domain.get(b, "Unknown"), **c}
            for m in model_names for b, c in results[m].items()]
    df = pd.DataFrame(rows)

    os.makedirs(f"{RESULTS}", exist_ok=True)
    out = "per_bin_counts.tsv" if eval_data_dir == DEFAULT_VARIANT else f"per_bin_counts_{eval_data_dir}.tsv"
    df.to_csv(f"{RESULTS}/{out}", sep="\t", index=False)
    print(f"{df['genome_bin_id'].nunique()} genome bins written to {RESULTS}/{out}")
    print("STEP 1 DONE")


if __name__ == "__main__":
    main()
