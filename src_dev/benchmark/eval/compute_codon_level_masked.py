"""Codon-level evaluation with short annotated CDS (<= 60 bp) MASKED, for all read-based test sets.

For every test set, test genome and model this computes, with the shared functions in codon_level_functions.py:
  - "original": codon counts with the original rules (short annotated CDS counted as non-coding), plus the
    explicit filter of predicted CDS <= 60 bp (removes nothing in the current predictions; counted and
    reported), and
  - "masked": the same, with codons of annotated CDS <= 60 bp excluded (mask from the raw read CSV, per RF).
60 bp reads are evaluated for the DeepCDS models only, with CDS fragments down to 30 bp (testset_dict_30.pkl,
model_preds_dict_30.pkl); there the threshold is 30 bp instead of 61 bp: annotated CDS < 30 bp are masked and
predicted CDS < 30 bp are filtered. All other read lengths: all models, > 60 bp.
Genomes are processed in parallel (the ERDA mount is slow per file but fast in parallel).

Outputs (under data/processed_data/report_results/, nothing existing is overwritten):
  - masked caches in the format the notebooks load (MASK_SHORT_CDS = True in the notebooks):
      without_errors/codon_level_read_length_masked/
      with_errors/codon_level_read_length_masked/{indel}i_{sub}s/
      with_errors/codon_level_sequencer_platforms_masked/      (ART; new pooled codon counts)
  - with_errors/codon_level_sequencer_platforms_codon/          (ART, original rules; the ART notebook had no
                                                                   pooled codon-level counts before)
  - codon_level_masking_comparison/<suite>.pkl: per test set, model and genome: original and masked counts,
    masked codons, removed short predictions (a check of the masking; not used for the results)

Usage: python compute_codon_level_masked.py --suite {without_errors,with_errors_typical,with_errors_high,
       with_errors_stress,art_modern,all} [--procs 48]
"""
import argparse
import os
import pickle
import sys
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import codon_level_functions as clf  # noqa: E402

project_root = "/tmp/nrt204/FragmentPredictor"
D = f"{project_root}/data/processed_data"
CKPT = "full_model_all_genomes_seed_42_trained_final_8M_no_dropout"
ERROR_LEVELS = {"typical": ("5e-06", "0.004"), "high": ("1.25e-05", "0.01"), "stress": ("3.75e-05", "0.03")}
MODELS_7 = ["fgs_complete", "fgs_illumina_5", "fgs_illumina_10", "prodigal",
            "deep_cds_no_errors", "deep_cds_substitution", "deep_cds_errors"]


def pred_path(suite, data_type, model, acc, preds_file="model_preds_dict.pkl"):
    """Prediction paths exactly as in each notebook's load_model_preds() (preds_file: _30 dict for 60 bp)."""
    return _pred_path(suite, data_type, model, acc).replace("model_preds_dict.pkl", preds_file)


def _pred_path(suite, data_type, model, acc):
    if suite == "without_errors":
        return {"prodigal": f"{D}/predictions/processed_predictions/prodigal_preds/{data_type}/{acc}/model_preds_dict.pkl",
                "fgs_complete": f"{D}/predictions/processed_predictions/fgs_preds/{data_type}/{acc}.pkl",
                "deep_cds": f"{D}/predictions/processed_predictions/DeepCDS/model_without_errors/{data_type}/{CKPT}/{acc}/model_preds_dict.pkl"}[model]
    return {"prodigal": f"{D}/predictions/processed_predictions/prodigal_preds/{data_type}/{acc}/model_preds_dict.pkl",
            "fgs_complete": f"{D}/predictions/processed_predictions/fgs_preds/{data_type}_complete/{acc}.pkl",
            "fgs_illumina_5": f"{D}/predictions/processed_predictions/fgs_preds/{data_type}_illumina_5/{acc}.pkl",
            "fgs_illumina_10": f"{D}/predictions/processed_predictions/fgs_preds/{data_type}_illumina_10/{acc}.pkl",
            "deep_cds_no_errors": f"{D}/predictions/processed_predictions/DeepCDS/model_without_errors/{data_type}/{CKPT}/{acc}/model_preds_dict.pkl",
            "deep_cds_substitution": f"{D}/predictions/processed_predictions/DeepCDS/model_with_substitution_errors/{data_type}/{CKPT}/{acc}/model_preds_dict.pkl",
            "deep_cds_errors": f"{D}/predictions/processed_predictions/DeepCDS/model_with_errors/{data_type}/{CKPT}/{acc}/model_preds_dict.pkl"}[model]


def _rules(read_length, models, deepcds_models):
    """(models, testset file, predictions file, max length of a 'short' CDS) for one read length."""
    if read_length == 60:  # DeepCDS only, fragments down to 30 bp
        return (deepcds_models, "testset_dict_30.pkl", "model_preds_dict_30.pkl", 29)
    return (models, "testset_dict.pkl", "model_preds_dict.pkl", clf.SHORT_CDS_MAX_BP)


def suite_config(suite):
    """(list of (key, data_type, seq_len), models, masked cache dir, original cache dir or None)."""
    if suite == "without_errors":
        models = ["fgs_complete", "prodigal", "deep_cds"]
        sets = [(L, f"without_errors_{L}bp", L) + _rules(L, models, ["deep_cds"]) for L in (60, 75, 100, 150, 300, 700, 1000)]
        return sets, models, \
            f"{D}/report_results/without_errors/codon_level_read_length_masked", None
    if suite.startswith("with_errors_"):
        i, s = ERROR_LEVELS[suite.split("_")[-1]]
        sets = [(L, f"with_errors_{i}i_{s}s_{L}bp", L) + _rules(L, MODELS_7, MODELS_7[4:]) for L in (60, 75, 100, 150, 300)]
        return sets, MODELS_7, f"{D}/report_results/with_errors/codon_level_read_length_masked/{i}i_{s}s", None
    if suite == "art_modern":
        sets = [(p, p, int(p.split("_")[-1][:-2])) + _rules(150, MODELS_7, MODELS_7[4:])
                for p in ("HiSeq2500_150bp", "NextSeq500_150bp", "MiSeq_v3_300bp")]
        return sets, MODELS_7, f"{D}/report_results/with_errors/codon_level_sequencer_platforms_masked", \
            f"{D}/report_results/with_errors/codon_level_sequencer_platforms_codon"
    raise ValueError(suite)


def evaluate_genome(args):
    suite, data_type, seq_len, acc, models, testset_file, preds_file, max_bp = args
    try:
        with open(f"{D}/testset_processed/{data_type}/{acc}/{testset_file}", "rb") as f:
            testset_dict = clf.get_full_cds(pickle.load(f))
        with open(f"{D}/testset_processed/{data_type}/{acc}/read_names_list.pkl", "rb") as f:
            read_names_list = pickle.load(f)
    except FileNotFoundError:
        return acc, None
    mask = clf.short_cds_mask_spans(f"{D}/reads_processed/test/{data_type}/csv/{acc}.csv.gz", max_bp=max_bp)
    codons_per_rf = (seq_len - 3) // 3
    codons_total = codons_per_rf * 3
    out = {}
    for model in models:
        try:
            with open(pred_path(suite, data_type, model, acc, preds_file), "rb") as f:
                preds = clf.get_full_cds(pickle.load(f))
        except FileNotFoundError:
            continue
        preds, n_removed = clf.filter_short_predictions(preds, max_bp=max_bp)
        orig = clf.calculate_performance_metrics_model(testset_dict, preds, read_names_list, codons_per_rf, codons_total)
        masked = clf.calculate_performance_metrics_model(testset_dict, preds, read_names_list, codons_per_rf,
                                                         codons_total, mask_spans=mask)
        out[model] = {"original": orig, "masked": masked, "short_predictions_removed": n_removed}
    return acc, {"models": out, "n_reads": len(read_names_list), "codons_total": codons_total * len(read_names_list),
                 "reads_with_mask": sum(1 for r in read_names_list if r in mask)}


def genome_info():
    accs = open(f"{D}/genome_partitions/test_partition_accessions.txt").read().splitlines()
    info = pd.read_csv(f"{D}/dataset_information/genomes_info_with_partitions.csv").set_index("accession")
    out = {}
    for a in accs:
        if a in info.index:
            out[a] = {"gc_content": info.loc[a, "gc_content"],
                      "translation_table": 4 if info.loc[a, "family"] == "Mycoplasmataceae" else 11}
    return accs, out


GC_BINS = [0, 40, 50, 60, 100]
GC_LABELS = [r"$<$40\%", r"40--50\%", r"50--60\%", r"$>$60\%"]


def gc_label(gc):
    for i in range(len(GC_BINS) - 1):
        if GC_BINS[i] <= gc < GC_BINS[i + 1]:
            return GC_LABELS[i]
    return None


def notebook_cache(results, keys, models, genomes, which):
    """Rebuild the read-length notebooks' cache objects from per-genome counts (same aggregation as their loop)."""
    z = lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    rbl = {k: {m: z() for m in models} for k in keys}
    gc_d = lambda: {k: {g: {m: z() for m in models} for g in GC_LABELS} for k in keys}
    rblgc, rblgc11, rblgc4 = gc_d(), gc_d(), gc_d()
    hd = lambda: {k: {m: False for m in models} for k in keys}
    has, has11, has4 = hd(), hd(), hd()
    pgml = {k: {m: {"mcc": [], "f1_score": [], "precision": [], "recall": []} for m in models} for k in keys}
    pgc = {k: {m: {} for m in models} for k in keys}
    for k in keys:
        for acc, res in results[k].items():
            if res is None:
                continue
            tt = genomes.get(acc, {}).get("translation_table")
            gl = gc_label(genomes[acc]["gc_content"]) if acc in genomes else None
            for m in models:
                if m not in res["models"]:
                    continue
                c = res["models"][m][which]
                cnt = {x: c[x] for x in ("tp", "fp", "fn", "tn")}
                if tt != 4:
                    for x in cnt:
                        rbl[k][m][x] += cnt[x]
                    has[k][m] = True
                    for x in ("mcc", "f1_score", "precision", "recall"):
                        pgml[k][m][x].append(c[x])
                    pgc[k][m][acc] = dict(cnt)
                    if gl is not None:
                        for x in cnt:
                            rblgc[k][gl][m][x] += cnt[x]
                if gl is not None and tt in (11, 4):
                    target, flag = (rblgc11, has11) if tt == 11 else (rblgc4, has4)
                    for x in cnt:
                        target[k][gl][m][x] += cnt[x]
                    flag[k][m] = True
    return {"results_by_length": rbl, "results_by_length_gc": rblgc, "has_data": has,
            "results_by_length_gc_tt11": rblgc11, "results_by_length_gc_tt4": rblgc4,
            "has_data_tt11": has11, "has_data_tt4": has4,
            "per_genome_metrics_list": pgml, "per_genome_counts": pgc}


def art_cache(results, keys, models, genomes, which):
    """ART notebook: pooled codon counts per platform + per-genome lists/counts (TT4 excluded)."""
    rbp = {k: {m: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for m in models} for k in keys}
    has = {k: {m: False for m in models} for k in keys}
    pgml = {k: {m: {"mcc": [], "f1_score": [], "precision": [], "recall": []} for m in models} for k in keys}
    pgc = {k: {m: {} for m in models} for k in keys}
    for k in keys:
        for acc, res in results[k].items():
            if res is None or genomes.get(acc, {}).get("translation_table") == 4:
                continue
            for m, r in res["models"].items():
                c = r[which]
                for x in ("tp", "fp", "fn", "tn"):
                    rbp[k][m][x] += c[x]
                has[k][m] = True
                for x in ("mcc", "f1_score", "precision", "recall"):
                    pgml[k][m][x].append(c[x])
                pgc[k][m][acc] = {x: c[x] for x in ("tp", "fp", "fn", "tn")}
    return {"codon_results_by_platform": rbp, "codon_has_data": has,
            "per_genome_codon_metrics_list": pgml, "per_genome_codon_counts": pgc}


def write(objs, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    for name, obj in objs.items():
        with open(f"{cache_dir}/{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
    print(f"  wrote {cache_dir}", flush=True)


def run_suite(suite, procs):
    sets, models, masked_dir, original_dir = suite_config(suite)
    accs, genomes = genome_info()
    results = {}
    with Pool(procs) as pool:
        for key, data_type, seq_len, set_models, testset_file, preds_file, max_bp in sets:
            res = pool.map(evaluate_genome, [(suite, data_type, seq_len, a, set_models, testset_file, preds_file, max_bp)
                                             for a in accs], chunksize=1)
            results[key] = dict(res)
            n_ok = sum(v is not None for v in results[key].values())
            removed = sum(r["short_predictions_removed"] for v in results[key].values() if v for r in v["models"].values())
            print(f"{suite} / {data_type}: {n_ok} genomes; short predictions removed by the filter: {removed}", flush=True)
    keys = [st[0] for st in sets]
    if suite == "art_modern":
        write(art_cache(results, keys, models, genomes, "masked"), masked_dir)
        write(art_cache(results, keys, models, genomes, "original"), original_dir)
    else:
        write(notebook_cache(results, keys, models, genomes, "masked"), masked_dir)
    os.makedirs(f"{D}/report_results/codon_level_masking_comparison", exist_ok=True)
    with open(f"{D}/report_results/codon_level_masking_comparison/{suite}.pkl", "wb") as f:
        pickle.dump({"sets": sets, "models": models, "results": results, "genomes": genomes}, f)
    print(f"  wrote comparison for {suite}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--procs", type=int, default=48)
    args = parser.parse_args()
    suites = (["without_errors", "with_errors_typical", "with_errors_high", "with_errors_stress", "art_modern"]
              if args.suite == "all" else [args.suite])
    for s in suites:
        run_suite(s, args.procs)
    print("ALL DONE")
