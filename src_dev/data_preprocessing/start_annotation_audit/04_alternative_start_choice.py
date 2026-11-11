"""Which start does each model choose where the annotation has a longer and a
shorter version of a CDS (same stop, different starts; step 01, the 6.6% of test CDS)?

The four 300 bp test sets (error-free, typical, high, stress), with the start definition of assessment A in
start_codon_by_codon_type.ipynb (notebook cells 1, 2, 4, 5; predicted starts = get_real_starts of the predicted CDS
fragments; the read's first in-frame codon excluded). Models: error-free as the notebook (FGS (Complete), MetaProdigal,
DeepCDS N); with errors also FGS (0.5% / 1.0% error rate), DeepCDS S and DeepCDS S+I, as in the benchmark.

For each same-stop pair (longest start L = kept_start, shortest start S = dropped_start; the longest CDS is the one in
the final annotation), both starts are placed in each test read (start_coord: position p of the read is genome position
start_coord + p - 1 for error-free reads; reads with errors are mapped through their CIGAR string, and a
start whose codon contains a deleted base or an insertion is not in the read; rf = (p - 1) % 3). A start counts as in the
read if its whole codon is in the read, it is not the read's first in-frame codon, and the CDS fragment it starts in the
read is longer than 60 bp (from the start to the stop codon, the read's last whole codon in that frame if the stop is
outside the read, or the base before the first indel after the start, where the labeled CDS fragment ends) - the same
> 60 bp rule as the ground truth and every model's predictions. Check: the longest start, where placed, must be a
labeled start in the ground truth (counted as longest_is_labeled_start).

Only reads containing BOTH L and S are counted, one case per read and pair:
  longest  (L predicted, S not), shortest (S predicted, L not), both, neither.
Pooled over the TT11 test genomes; ratio longest / shortest with a genome-level cluster bootstrap SE (N = 1000,
seed 42).
Output: RES/alternative_start_choice.tsv
Usage: python 04_alternative_start_choice.py [--procs 64]
"""
import argparse
import json
import os
import re
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "/tmp/nrt204/FragmentPredictor/data/processed_data/start_annotation_audit"
NB_DIR = os.path.abspath(f"{HERE}/../../benchmark/eval/without_errors")
NB = json.load(open(f"{NB_DIR}/start_codon_by_codon_type.ipynb"))
G = {"__name__": "__nbsetup__"}
os.chdir(NB_DIR)
for i in (1, 2, 4, 5):
    exec("".join(NB["cells"][i]["source"]), G)
os.chdir(HERE)
READ_LENGTH = 300
MIN_FRAGMENT = 61  # > 60 bp
TEST_SETS = {"Error-free": "without_errors", "Typical error": "with_errors_5e-06i_0.004s",
             "High error": "with_errors_1.25e-05i_0.01s", "Stress error": "with_errors_3.75e-05i_0.03s"}
MODELS_ERROR_FREE = ["fgs_complete", "prodigal", "deep_cds_no_errors"]
MODELS_ERRORS = ["fgs_complete", "fgs_illumina_5", "fgs_illumina_10", "prodigal", "deep_cds_no_errors",
                 "deep_cds_substitution", "deep_cds_errors"]
DEEPCDS_RUN = "full_model_all_genomes_seed_42_trained_final_8M_no_dropout"
COUNTS = ["n", "longest", "shortest", "both", "neither"]
PAIRS = None


def init(pairs):
    global PAIRS
    PAIRS = pairs


def load_preds(acc, data_type, model):
    pp = f"{G['project_root']}/data/processed_data/predictions/processed_predictions"
    fgs = data_type if data_type.startswith("without_errors") else f"{data_type}_complete"
    path = {"prodigal": f"{pp}/prodigal_preds/{data_type}/{acc}/model_preds_dict.pkl",
            "fgs_complete": f"{pp}/fgs_preds/{fgs}/{acc}.pkl",
            "fgs_illumina_5": f"{pp}/fgs_preds/{data_type}_illumina_5/{acc}.pkl",
            "fgs_illumina_10": f"{pp}/fgs_preds/{data_type}_illumina_10/{acc}.pkl",
            "deep_cds_no_errors": f"{pp}/DeepCDS/model_without_errors/{data_type}/{DEEPCDS_RUN}/{acc}/model_preds_dict.pkl",
            "deep_cds_substitution": f"{pp}/DeepCDS/model_with_substitution_errors/{data_type}/{DEEPCDS_RUN}/{acc}/model_preds_dict.pkl",
            "deep_cds_errors": f"{pp}/DeepCDS/model_with_errors/{data_type}/{DEEPCDS_RUN}/{acc}/model_preds_dict.pkl"}[model]
    return pd.read_pickle(path)


def cigar_map(cigar):
    """(genome offset -> read position, both 1-based relative to the read start; read positions of indels).
    Deleted genome bases are absent from the map; an insertion is recorded at the read position of the inserted base,
    a deletion at the read position after it."""
    g, r, m, indels = 0, 0, {}, []
    for n, op in re.findall(r"(\d+)([MIDNSX=])", cigar):
        n = int(n)
        if op in "M=X":
            for _ in range(n):
                g, r = g + 1, r + 1
                m[g] = r
        elif op in "IS":
            indels += [r + k + 1 for k in range(n)] if op == "I" else []
            r += n
        elif op in "DN":
            indels.append(r + 1)
            g += n
    return m, sorted(indels)


def to_read(pos, c, read_len, stop_end, gmap=None, indels=()):
    """(start, rf) of a genome-coordinate codon start in a read, or None if outside it, at the border (A), hit by an
    indel, or if the CDS fragment it starts in the read is not longer than 60 bp."""
    if gmap is None:
        s, stop = pos - c + 1, stop_end - c + 1
    else:
        codon = [gmap.get(pos - c + 1 + k) for k in range(3)]
        if None in codon or codon[2] - codon[0] != 2:
            return None
        s = codon[0]
        stop = gmap.get(stop_end - c + 1, read_len + 1)
    if s < 1 or s + 2 > read_len:
        return None
    rf = str((s - 1) % 3)
    if G["is_internal_start"](s, rf):
        return None
    end = min(stop, s + 3 * ((read_len - s + 1) // 3) - 1)  # stop codon, or last whole codon
    nxt = [i for i in indels if i > s]
    if nxt:
        end = min(end, nxt[0] - 1)
    if end - s + 1 < MIN_FRAGMENT:
        return None
    return (s, rf)


def genome(task):
    test_set, acc = task
    data_type = f"{TEST_SETS[test_set]}_{READ_LENGTH}bp"
    errors = test_set != "Error-free"
    read_names_list, testset_dict = G["load_testset"](acc, data_type)
    cols = ["read_name", "read", "start_coord", "assembly_id", "strand"] + (["CIGAR"] if errors else [])
    df = pd.read_csv(f"{G['project_root']}/data/processed_data/reads_processed/test/{data_type}/csv/{acc}.csv.gz",
                     usecols=cols)
    info = {r.read_name: r for r in df.itertuples()}

    # Per read: [(L, S)] in read coordinates, for every pair with both starts in the read
    placed, n_longest_labeled = {}, 0
    for name in read_names_list:
        r = info.get(name)
        if r is None:
            continue
        gmap, indels = cigar_map(r.CIGAR) if errors else (None, ())
        hits = []
        for k, d, stop_end in PAIRS.get((acc, r.assembly_id, r.strand), []):
            lk = to_read(k, r.start_coord, len(r.read), stop_end, gmap, indels)
            ld = to_read(d, r.start_coord, len(r.read), stop_end, gmap, indels)
            if lk and ld:
                hits.append((lk, ld))
        if hits:
            placed[name] = hits
            truth = G["get_real_starts"](testset_dict.get(name, {}).get("cds_coords", []))
            n_longest_labeled += sum(lk in truth for lk, _ in hits)

    models = MODELS_ERRORS if errors else MODELS_ERROR_FREE
    out = {m: dict.fromkeys(COUNTS, 0) for m in models}
    for model in models:
        preds = load_preds(acc, data_type, model)
        o = out[model]
        for name, hits in placed.items():
            pred = G["get_real_starts"](preds.get(name, {}).get("cds_coords", []))
            for lk, ld in hits:
                pk, pd_ = lk in pred, ld in pred
                o["n"] += 1
                o["both" if pk and pd_ else "longest" if pk else "shortest" if pd_ else "neither"] += 1
    return task, out, n_longest_labeled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--procs", type=int, default=64)
    a = p.parse_args()
    groups = pd.read_csv(f"{RES}/start_conflict_groups.tsv", sep="\t")
    groups = groups[(groups["partition"] == "test") & groups["kept_in_final"]]
    pairs = defaultdict(list)
    for r in groups.itertuples():
        pairs[(r.accession, r.assembly, r.strand)].append((r.kept_start, r.dropped_start, r.stop_end))
    accs = [x for x in G["all_test_accessions"] if G["all_genomes"].get(x, {}).get("translation_table") != 4]
    tasks = [(t, x) for t in TEST_SETS for x in accs]
    with Pool(a.procs, initializer=init, initargs=(dict(pairs),)) as pool:
        res = pool.map(genome, tasks, chunksize=1)

    rows = []
    for test_set in TEST_SETS:
        rs = [(out, nl) for (t, _), out, nl in res if t == test_set]
        rng = np.random.default_rng(42)
        idx = rng.integers(0, len(rs), size=(1000, len(rs)))
        for m in rs[0][0]:
            c = {k: np.array([out[m][k] for out, _ in rs], dtype=float) for k in COUNTS}
            row = {"test_set": test_set, "read_length": READ_LENGTH, "model": G["model_display_names"][m],
                   **{k: int(c[k].sum()) for k in COUNTS}}
            for k in ("longest", "shortest", "both", "neither"):
                row[f"{k}_pct"] = 100 * c[k].sum() / c["n"].sum()
            row["ratio_longest_shortest"] = c["longest"].sum() / c["shortest"].sum()
            row["ratio_longest_shortest_se"] = float(np.std(c["longest"][idx].sum(axis=1) / c["shortest"][idx].sum(axis=1),
                                                            ddof=1))
            row["longest_is_labeled_start"] = sum(nl for _, nl in rs)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(f"{RES}/alternative_start_choice.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
