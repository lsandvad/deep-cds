"""Shared codon-level evaluation functions for the read-based benchmarks.

Used by eval/without_errors/codon_level_read_length.ipynb, eval/with_errors/codon_level_read_length.ipynb,
eval/art_modern/cds_and_codon_level_performance.ipynb and eval/compute_codon_level_masked.py.

Two rules, identical for every model:

1. MASKING (ground truth only). Codons belonging to an annotated CDS that is <= 60 bp long are excluded from
   the evaluation (not TP, FP, FN or TN). "Length" is the CDS span exactly as in postprocess_testset.py: for
   CDS split by indels, from the first piece's start to the last piece's end over the connected pieces
   (cds_fragments_connection). The mask is built per read and per reading frame from ALL labels in the raw
   read CSV (reads_processed/test/<set>/csv/<acc>.csv.gz): CDS < 30 bp are not in any testset pickle.
   Each piece is masked in the reading frame in which it is annotated.

2. FILTERING PREDICTIONS. Predicted CDS with a span <= 60 bp are removed from a model's predictions, so their
   codons count as predicted NON-coding; positions are never dropped because of a prediction. (Prodigal and
   DeepCDS postprocessing already remove them; FragGeneScan does not output them. The explicit filter makes
   the rule identical and visible for all models.)

With mask_spans=None the counts are identical to the original notebooks' calculate_performance_metrics_model.
"""
import ast
import math

import numpy as np
import pandas as pd

SHORT_CDS_MAX_BP = 60


# ---------------------------------------------------------------------------------------------------------
# Copied from eval/with_errors/codon_level_read_length.ipynb (unchanged)
# ---------------------------------------------------------------------------------------------------------
def get_full_cds(reads_dict):
    """Merge fragmented CDS annotations that belong to the same gene (indel-disrupted CDS).

    The merged CDS is labelled 'disrupted_rf' and keeps the original per-fragment [start, end, rf] entries as
    a 4th element so codon-level evaluation can assign each sub-span its correct RF.
    """
    for read in reads_dict.keys():
        cds_fragments_connections = reads_dict[read]['cds_fragments_connection']
        cds_coords = reads_dict[read]['cds_coords']

        indices_to_remove = []

        for cds_frag_mapping in cds_fragments_connections:
            if len(cds_frag_mapping) > 1:
                if len(cds_fragments_connections) != cds_frag_mapping[0] + 1:
                    if len(cds_frag_mapping) == 1:
                        cds_frag_mapping = [cds_index - 1 for cds_index in cds_frag_mapping]

                elif len(cds_fragments_connections) == 1 and cds_frag_mapping == [0, 2]:
                    cds_frag_mapping = [0, 1]
                elif len(cds_fragments_connections) == 1 and cds_frag_mapping == [2, 3, 4]:
                    cds_frag_mapping = [0, 1, 2]

                original_fragments = [list(cds_coords[i]) for i in sorted(cds_frag_mapping)]

                start_cds_frag = cds_coords[min(cds_frag_mapping)][0]
                end_cds_frag = cds_coords[max(cds_frag_mapping)][1]

                indices_to_remove += cds_frag_mapping

                assert start_cds_frag < end_cds_frag, "Something went wrong processing indel-induced error in sequence!"
                cds_coords.append([start_cds_frag, end_cds_frag, 'disrupted_rf', original_fragments])
                reads_dict[read]['cds_coords'] = cds_coords

        for index in sorted(indices_to_remove, reverse=True):
            cds_coords.pop(index)

    return reads_dict


def split_by_rf(cds_coords):
    """Split CDS coordinates into per-RF lists of (start, end), normalised to frame 0 (rf1 -1, rf2 -2).
    'disrupted_rf' entries are expanded into their original fragments."""
    rf0, rf1, rf2 = [], [], []
    for c in cds_coords:
        if c[2] == 'disrupted_rf':
            for frag in c[3]:
                s, e, rf = frag[0], frag[1], frag[2]
                if rf == '0':
                    rf0.append((s, e))
                elif rf == '1':
                    rf1.append((s - 1, e - 1))
                elif rf == '2':
                    rf2.append((s - 2, e - 2))
        elif c[2] == '0':
            rf0.append((c[0], c[1]))
        elif c[2] == '1':
            rf1.append((c[0] - 1, c[1] - 1))
        elif c[2] == '2':
            rf2.append((c[0] - 2, c[1] - 2))
    return rf0, rf1, rf2


def matthews_corrcoef(TP, FP, FN, TN):
    numerator = (TP * TN) - (FP * FN)
    denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return numerator / denominator if denominator != 0 else 0.0


# ---------------------------------------------------------------------------------------------------------
# Rule 2: predictions <= 60 bp -> non-coding
# ---------------------------------------------------------------------------------------------------------
def filter_short_predictions(preds_dict, max_bp=SHORT_CDS_MAX_BP):
    """Drop predicted CDS whose span (start to end, incl. all pieces of a disrupted CDS) is <= max_bp.
    Apply AFTER get_full_cds(). Returns (filtered_dict, n_removed)."""
    out, n_removed = {}, 0
    for read, entry in preds_dict.items():
        kept = [c for c in entry['cds_coords'] if c[1] - c[0] + 1 > max_bp]
        n_removed += len(entry['cds_coords']) - len(kept)
        if kept:
            out[read] = {**entry, 'cds_coords': kept}
    return out, n_removed


# ---------------------------------------------------------------------------------------------------------
# Rule 1: mask of annotated CDS <= 60 bp, from the raw read CSV
# ---------------------------------------------------------------------------------------------------------
def _parse(value, default):
    if isinstance(value, str) and value.strip() not in ("", "[]"):
        return ast.literal_eval(value)
    return default


def short_cds_mask_spans(reads_csv_path, max_bp=SHORT_CDS_MAX_BP):
    """{read_name: (rf0, rf1, rf2)} with the frame-0-normalised (start, end) spans of every annotated CDS whose
    span is <= max_bp, in the reading frame of each of its pieces. Reads without short CDS are absent.

    Pieces are grouped by cds_fragments_connection when present (indel test sets), otherwise each CDS is its
    own group - the same grouping postprocess_testset.py uses to decide what goes into testset_dict.pkl."""
    df = pd.read_csv(reads_csv_path, compression="infer", index_col=0,
                     usecols=lambda c: c in ("Unnamed: 0", "read_name", "cds_coords", "cds_fragments_connection"))
    has_conn = "cds_fragments_connection" in df.columns
    mask = {}
    for row in df.itertuples(index=False):
        coords = _parse(row.cds_coords, [])
        if not coords:
            continue
        groups = _parse(row.cds_fragments_connection, None) if has_conn else None
        if not groups:
            groups = [[i] for i in range(len(coords))]
        short = []
        for g in groups:
            pieces = [coords[i] for i in g]
            span = max(p[1] for p in pieces) - min(p[0] for p in pieces) + 1
            if span <= max_bp:
                short.extend(pieces)
        if short:
            mask[row.read_name] = split_by_rf([[s, e, str(rf)] for s, e, rf in short])
    return mask


# ---------------------------------------------------------------------------------------------------------
# Codon counting (original logic + optional mask)
# ---------------------------------------------------------------------------------------------------------
def calculate_rates_for_rf(cds_rf_testset, cds_rf_preds, codons_per_rf, mask_rf=()):
    """Codon-level TP/FP/FN/TN for one reading frame (frame-0-normalised, 1-based coordinates), plus the
    number of masked codons. Masked positions are excluded from all four counts."""
    seq_len_pred = codons_per_rf * 3  # truncate to nearest multiple of 3
    pred_covered = np.zeros(seq_len_pred, dtype=bool)
    actual_covered = np.zeros(seq_len_pred, dtype=bool)
    keep = np.ones(seq_len_pred, dtype=bool)

    for start, end in cds_rf_preds:
        pred_covered[start - 1:end] = True
    for start, end in cds_rf_testset:
        actual_covered[start - 1:end] = True
    for start, end in mask_rf:
        keep[max(start - 1, 0):end] = False

    counts = [int(np.sum(pred_covered & actual_covered & keep)), int(np.sum(pred_covered & ~actual_covered & keep)),
              int(np.sum(~pred_covered & actual_covered & keep)), int(np.sum(~pred_covered & ~actual_covered & keep))]
    masked_bp = seq_len_pred - int(keep.sum())
    assert masked_bp % 3 == 0, "mask is not codon-aligned"
    TP, FP, FN, TN = (int(c / 3) for c in counts)
    masked = masked_bp // 3
    assert TP + FP + FN + TN + masked == codons_per_rf, (TP, FP, FN, TN, masked, codons_per_rf)
    return TP, FP, FN, TN, masked


def calculate_performance_metrics_model(testset_dict, model_preds_dict, read_names_list,
                                        codons_per_rf, codons_total, seq_len=None, mask_spans=None):
    """Codon-level evaluation of one model over all reads. mask_spans: output of short_cds_mask_spans, or None.
    Returns tp/fp/fn/tn/masked counts and mcc/f1/precision/recall. 'masked' is the same for every model."""
    TP = FP = FN = TN = MASKED = 0
    empty = ([], [], [])
    mask_spans = mask_spans or {}

    for read in read_names_list:
        in_test = read in testset_dict
        in_pred = read in model_preds_dict
        read_mask = mask_spans.get(read)

        if not in_test and not in_pred and read_mask is None:
            TN += codons_total
            continue
        t = split_by_rf(testset_dict[read]['cds_coords']) if in_test else empty
        p = split_by_rf(model_preds_dict[read]['cds_coords']) if in_pred else empty
        m = read_mask if read_mask is not None else empty
        for rf_t, rf_p, rf_m in zip(t, p, m):
            tp_, fp_, fn_, tn_, ms_ = calculate_rates_for_rf(rf_t, rf_p, codons_per_rf, rf_m)
            TP += tp_; FP += fp_; FN += fn_; TN += tn_; MASKED += ms_

    mcc = matthews_corrcoef(TP, FP, FN, TN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'tp': TP, 'fp': FP, 'fn': FN, 'tn': TN, 'masked': MASKED,
            'mcc': round(mcc, 4), 'f1_score': round(f1, 4),
            'precision': round(precision, 4), 'recall': round(recall, 4)}


# ---------------------------------------------------------------------------------------------------------
# Genome-level cluster bootstrap (as in the codon-level notebooks; N=1000, seed 42)
# ---------------------------------------------------------------------------------------------------------
def pooled_scores(tp, fp, fn, tn):
    """Pooled MCC/precision/recall/F1 from summed counts; array-safe, 0 where undefined."""
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(tp, dtype=float), where=(p + r) > 0)
    mcc_num = tp * tn - fp * fn
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(mcc_num, mcc_den, out=np.zeros_like(tp, dtype=float), where=mcc_den > 0)
    return {'mcc': mcc, 'precision': p, 'recall': r, 'f1_score': f1}


def bootstrap_pooled_std(genome_counts_dict, n_boot=1000, rng=None):
    """SE of the pooled scores by resampling genomes with replacement (sum raw counts per replicate)."""
    rng = rng if rng is not None else np.random.default_rng(42)
    accs = list(genome_counts_dict)
    if len(accs) < 2:
        return {k: np.nan for k in ('mcc', 'precision', 'recall', 'f1_score')}
    arr = {k: np.array([genome_counts_dict[a][k] for a in accs], dtype=float) for k in ('tp', 'fp', 'fn', 'tn')}
    idx = rng.integers(0, len(accs), size=(n_boot, len(accs)))
    boot = pooled_scores(*(arr[k][idx].sum(axis=1) for k in ('tp', 'fp', 'fn', 'tn')))
    return {k: float(v.std(ddof=1)) for k, v in boot.items()}
