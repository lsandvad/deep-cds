"""Compute the cache for start_codon_by_codon_type.ipynb.

Runs the notebook's own setup, loader and scoring cells (cells 1, 2, 4, 5: loaders, get_real_starts,
split_by_codon_type, start_counts_by_codon_type, ...) and the per-genome body of its main loop (cell 7), in parallel
over (read length, genome) - the ERDA mount is slow per file but fast in parallel. Accumulation and the cached objects
are identical to the notebook's serial loop. Usage: python start_codon_by_codon_type_compute.py [--procs 48]
"""
import argparse
import json
import os
import pickle
from collections import Counter
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
NB = json.load(open(os.path.join(HERE, "start_codon_by_codon_type.ipynb")))
G = {"__name__": "__nbsetup__"}
os.chdir(HERE)
for i in (1, 2, 4, 5):
    exec("".join(NB["cells"][i]["source"]), G)
codon_types, read_lengths = G["codon_types"], G["read_lengths"]
model_names, border_inclusive_models = G["model_names"], G["border_inclusive_models"]


def new_counts():
    return {"tp": 0, "fp": 0, "fn": 0}


def genome(task):
    """One (read length, genome): the body of the notebook's main loop, returning its contributions."""
    length, acc = task
    data_type = f"without_errors_{length}bp"
    g = G
    try:
        read_names_list, testset_dict = g["load_testset"](acc, data_type)
        read_seqs = g["load_read_sequences"](acc, data_type)
        read_labels = g["load_read_labels"](acc, data_type)
    except Exception as e:
        return task, {"error": f"testset/reads: {e}"}
    out = {"actual_excl": Counter(), "actual_incl": Counter(), "border": Counter(), "non_atg": Counter(),
           "excl": {}, "incl": {}, "border_only": {}, "errors": []}
    truth_excl, truth_incl = {}, {}
    for read_name in read_names_list:
        actual = testset_dict.get(read_name, {}).get("cds_coords", [])
        read = read_seqs.get(read_name)
        if read is None or not actual:
            continue
        labels_by_rf = read_labels.get(read_name, {})
        s_excl = g["get_real_starts"](actual)
        s_incl = g["get_real_starts"](actual, labels_by_rf)
        truth_excl[read_name], truth_incl[read_name] = s_excl, s_incl
        for key, starts in (("actual_excl", s_excl), ("actual_incl", s_incl)):
            for ct, ss in g["split_by_codon_type"](starts, read).items():
                out[key][ct] += len(ss)
        for ct, ss in g["split_by_codon_type"](s_incl - s_excl, read).items():
            out["border"][ct] += len(ss)
        for start, rf in s_incl:
            if g["get_start_codon"](read, start) != "ATG":
                out["non_atg"][g["get_start_codon"](read, start)] += 1
    for model in model_names:  # assessment A
        try:
            preds = g["load_model_preds"](acc, data_type, model)
        except Exception as e:
            out["errors"].append(f"{model}: {e}")
            continue
        pg = {ct: new_counts() for ct in codon_types}
        for read_name in read_names_list:
            read = read_seqs.get(read_name)
            if read is None:
                continue
            pred_starts = g["get_real_starts"](preds.get(read_name, {}).get("cds_coords", []))
            actual_starts = truth_excl.get(read_name, set())
            if not pred_starts and not actual_starts:
                continue
            for ct, (tp, fp, fn) in g["start_counts_by_codon_type"](pred_starts, actual_starts, read).items():
                pg[ct]["tp"] += tp; pg[ct]["fp"] += fp; pg[ct]["fn"] += fn
        out["excl"][model] = pg
    for model in border_inclusive_models:  # assessment B
        try:
            claimed = g["CLAIMED_START_LOADERS"][model](acc, data_type)
        except Exception as e:
            out["errors"].append(f"raw {model}: {e}")
            continue
        pg = {ct: new_counts() for ct in codon_types}
        pgb = {ct: new_counts() for ct in codon_types}
        for read_name in read_names_list:
            read = read_seqs.get(read_name)
            if read is None:
                continue
            pred_starts = claimed.get(read_name, set())
            actual_starts = truth_incl.get(read_name, set())
            if not pred_starts and not actual_starts:
                continue
            for ct, (tp, fp, fn) in g["start_counts_by_codon_type"](pred_starts, actual_starts, read).items():
                pg[ct]["tp"] += tp; pg[ct]["fp"] += fp; pg[ct]["fn"] += fn
            b_pred = {s for s in pred_starts if g["is_border_start"](*s)}
            b_actual = {s for s in actual_starts if g["is_border_start"](*s)}
            if b_pred or b_actual:
                for ct, (tp, fp, fn) in g["start_counts_by_codon_type"](b_pred, b_actual, read).items():
                    pgb[ct]["tp"] += tp; pgb[ct]["fp"] += fp; pgb[ct]["fn"] += fn
        out["incl"][model], out["border_only"][model] = pg, pgb
    return task, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--procs", type=int, default=48)
    args = parser.parse_args()
    accs = [a for a in G["all_test_accessions"] if G["all_genomes"].get(a, {}).get("translation_table") != 4]
    tasks = [(L, a) for L in read_lengths for a in accs]
    print(f"{len(accs)} TT11 genomes x {len(read_lengths)} read lengths = {len(tasks)} tasks", flush=True)

    results_excl = {L: {m: {ct: new_counts() for ct in codon_types} for m in model_names} for L in read_lengths}
    per_genome_counts_excl = {L: {m: {ct: {} for ct in codon_types} for m in model_names} for L in read_lengths}
    has_data_excl = {L: {m: False for m in model_names} for L in read_lengths}
    actual_start_counts_excl = {L: {ct: 0 for ct in codon_types} for L in read_lengths}
    results_incl = {L: {m: {ct: new_counts() for ct in codon_types} for m in border_inclusive_models} for L in read_lengths}
    per_genome_counts_incl = {L: {m: {ct: {} for ct in codon_types} for m in border_inclusive_models} for L in read_lengths}
    has_data_incl = {L: {m: False for m in border_inclusive_models} for L in read_lengths}
    actual_start_counts_incl = {L: {ct: 0 for ct in codon_types} for L in read_lengths}
    border_start_counts = {L: {ct: 0 for ct in codon_types} for L in read_lengths}
    border_only_counts = {L: {m: {ct: new_counts() for ct in codon_types} for m in border_inclusive_models} for L in read_lengths}
    non_atg_codon_identity_counts = {L: Counter() for L in read_lengths}

    done = 0
    with Pool(args.procs) as pool:
        for (L, acc), out in pool.imap_unordered(genome, tasks):
            done += 1
            if "error" in out:
                print(f"  {L}bp {acc}: {out['error']}", flush=True)
                continue
            for e in out["errors"]:
                print(f"  {L}bp {acc}: {e}", flush=True)
            for ct in codon_types:
                actual_start_counts_excl[L][ct] += out["actual_excl"][ct]
                actual_start_counts_incl[L][ct] += out["actual_incl"][ct]
                border_start_counts[L][ct] += out["border"][ct]
            non_atg_codon_identity_counts[L].update(out["non_atg"])
            for m, pg in out["excl"].items():
                has_data_excl[L][m] = True
                for ct in codon_types:
                    per_genome_counts_excl[L][m][ct][acc] = dict(pg[ct])
                    for k in ("tp", "fp", "fn"):
                        results_excl[L][m][ct][k] += pg[ct][k]
            for m, pg in out["incl"].items():
                has_data_incl[L][m] = True
                for ct in codon_types:
                    per_genome_counts_incl[L][m][ct][acc] = dict(pg[ct])
                    for k in ("tp", "fp", "fn"):
                        results_incl[L][m][ct][k] += pg[ct][k]
                        border_only_counts[L][m][ct][k] += out["border_only"][m][ct][k]
            if done % 100 == 0:
                print(f"  {done}/{len(tasks)} done", flush=True)

    cache = G["_cache_dir"]
    os.makedirs(cache, exist_ok=True)
    objs = dict(results_excl=results_excl, per_genome_counts_excl=per_genome_counts_excl, has_data_excl=has_data_excl,
                actual_start_counts_excl=actual_start_counts_excl, results_incl=results_incl,
                per_genome_counts_incl=per_genome_counts_incl, has_data_incl=has_data_incl,
                actual_start_counts_incl=actual_start_counts_incl, border_start_counts=border_start_counts,
                border_only_counts=border_only_counts, non_atg_codon_identity_counts=non_atg_codon_identity_counts)
    for name, obj in objs.items():
        with open(os.path.join(cache, f"{name}.pkl"), "wb") as f:
            pickle.dump(obj, f)
    print(f"Results saved to {cache}")
    for L in read_lengths:
        print(L, "A truth:", actual_start_counts_excl[L], "| B truth:", actual_start_counts_incl[L])


if __name__ == "__main__":
    main()
