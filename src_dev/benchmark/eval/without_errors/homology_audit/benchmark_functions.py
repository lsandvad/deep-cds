"""CDS-level benchmark functions, copied VERBATIM from
src_dev/benchmark/eval/with_errors/cds_level_read_length.ipynb (cells 3 and 7) so that TP/FN
are counted exactly as in the manuscript: exact coordinate match (IoU = 1).

Do not edit the copied functions; if the notebook changes, re-copy them.
"""

import numpy as np

N_BOOTSTRAP = 1000
_rng = np.random.default_rng(42)


def get_full_cds(reads_dict):
    """Merge fragmented CDS annotations that belong to the same gene.

    When a CDS is split into multiple fragments (e.g. due to short fragment
    removal during preprocessing), this function reconnects them into a single
    CDS span using the cds_fragments_connection mapping. The merged CDS is
    labelled 'disrupted_rf' and replaces the individual fragments.

    Args:
        reads_dict: Dict mapping read names to dicts with 'cds_coords' (list of
            [start, end, rf] triplets) and 'cds_fragments_connection' (list of
            index groups indicating which fragments belong together).

    Returns:
        The modified reads_dict with fragmented CDS entries merged.
    """
    for read in reads_dict.keys():
        cds_fragments_connections = reads_dict[read]['cds_fragments_connection']
        cds_coords = reads_dict[read]['cds_coords']

        indices_to_remove = []
        
        for cds_frag_mapping in cds_fragments_connections:
            if len(cds_frag_mapping) > 1:
                # Postprocessing for testset due to how short fragments were removed
                if len(cds_fragments_connections) != cds_frag_mapping[0] + 1:
                    if len(cds_frag_mapping) == 1:
                        cds_frag_mapping = [cds_index - 1 for cds_index in cds_frag_mapping]
                
                elif len(cds_fragments_connections) == 1 and cds_frag_mapping == [0, 2]:
                    cds_frag_mapping = [0, 1]
                elif len(cds_fragments_connections) == 1 and cds_frag_mapping == [2, 3, 4]:
                    cds_frag_mapping = [0, 1, 2]
                
                start_cds_frag = cds_coords[min(cds_frag_mapping)][0]
                end_cds_frag = cds_coords[max(cds_frag_mapping)][1]

                indices_to_remove += cds_frag_mapping
                
                assert start_cds_frag < end_cds_frag, "Something went wrong processing indel-induced error in sequence!"
                cds_coords.append([start_cds_frag, end_cds_frag, 'disrupted_rf'])
                reads_dict[read]['cds_coords'] = cds_coords

        for index in sorted(indices_to_remove, reverse=True):
            cds_coords.pop(index)
    
    return reads_dict


def calculate_iou(coord1, coord2):
    """Calculate Intersection over Union (IoU) between two CDS coordinate spans.

    Only computes overlap for CDS on the same reading frame. Returns 0.0 if the
    reading frames differ or the spans do not overlap.

    Args:
        coord1: Tuple of (start, stop, reading_frame) for the first CDS.
        coord2: Tuple of (start, stop, reading_frame) for the second CDS.

    Returns:
        IoU value between 0.0 and 1.0.
    """
    start1, stop1, rf1 = coord1
    start2, stop2, rf2 = coord2
    
    if rf1 != rf2:
        return 0.0
    
    intersection_start = max(start1, start2)
    intersection_stop = min(stop1, stop2)
    
    if intersection_start >= intersection_stop:
        return 0.0
    
    intersection_length = intersection_stop - intersection_start
    
    union_start = min(start1, start2)
    union_stop = max(stop1, stop2)
    union_length = union_stop - union_start
    
    iou = intersection_length / union_length if union_length > 0 else 0.0
    return iou


def find_matching_genes(predicted_coords, actual_coords):
    """Match predicted CDS to ground-truth CDS requiring exact coordinate identity (IoU = 1).

    Each predicted CDS is matched to the best-overlapping unmatched ground-truth
    CDS, provided the IoU equals 1.0 (identical start, stop, and reading frame).

    Args:
        predicted_coords: List of [start, stop, rf] for predicted CDS.
        actual_coords: List of [start, stop, rf] for ground-truth CDS.

    Returns:
        Tuple of (matched_pairs, num_matches) where matched_pairs is a list of
        (predicted_idx, actual_idx) tuples.
    """
    matched_pairs = []
    used_actual = set()
    used_predicted = set()
    
    for pred_idx, pred_coord in enumerate(predicted_coords):
        best_iou = 0.0
        best_actual_idx = -1
        
        for actual_idx, actual_coord in enumerate(actual_coords):
            if actual_idx in used_actual:
                continue
                
            iou = calculate_iou(pred_coord, actual_coord)
            
            if iou == 1.0 and iou > best_iou:
                best_iou = iou
                best_actual_idx = actual_idx
        
        if best_actual_idx != -1:
            matched_pairs.append((pred_idx, best_actual_idx))
            used_actual.add(best_actual_idx)
            used_predicted.add(pred_idx)
    
    return matched_pairs, len(matched_pairs)


def calculate_metrics(testset_dict, model_preds_dict, all_test_read_names):
    """Calculate precision, recall, and F1 score at the CDS level.

    Iterates over all reads, matches predicted CDS to ground-truth CDS using
    exact coordinate matching (IoU = 1.0), and aggregates TP/FP/FN counts
    across all reads.

    Args:
        testset_dict: Dict mapping read names to ground-truth annotations with 'cds_coords'.
        model_preds_dict: Dict mapping read names to model predictions with 'cds_coords'.
        all_test_read_names: List of all read names to evaluate.

    Returns:
        Dict with keys: total_predicted_genes, total_actual_genes,
        total_true_positives, total_false_positives, total_false_negatives,
        precision, recall, f1_score.
    """
    total_tp = 0
    total_predicted_genes = 0
    total_actual_genes = 0
    
    for read_name in all_test_read_names:
        cds_coords_testset = []
        cds_coords_model = []
        
        if read_name in testset_dict:
            cds_coords_testset = testset_dict[read_name].get("cds_coords", [])
        
        if read_name in model_preds_dict:
            cds_coords_model = model_preds_dict[read_name].get("cds_coords", [])
        
        num_predicted = len(cds_coords_model)
        num_actual = len(cds_coords_testset)

        _, tp_count = find_matching_genes(cds_coords_model, cds_coords_testset)

        total_predicted_genes += num_predicted
        total_actual_genes += num_actual
        total_tp += tp_count
    
    total_fp = total_predicted_genes - total_tp
    total_fn = total_actual_genes - total_tp
    
    precision = total_tp / total_predicted_genes if total_predicted_genes > 0 else 0.0
    recall = total_tp / total_actual_genes if total_actual_genes > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    results = {
        'total_predicted_genes': total_predicted_genes,
        'total_actual_genes': total_actual_genes,
        'total_true_positives': total_tp,
        'total_false_positives': total_fp,
        'total_false_negatives': total_fn,
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1_score': round(f1_score, 3)
    }
    
    return results


def pooled_scores(tp, fp, fn):
    """Pooled precision/recall/F1 from summed counts; array-safe, 0 where undefined."""
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(tp, dtype=float), where=(p + r) > 0)
    return {'precision': p, 'recall': r, 'f1_score': f1}

def bootstrap_pooled_std(genome_counts_dict, n_boot=N_BOOTSTRAP, rng=_rng):
    """Cluster-bootstrap SE of the pooled scores, resampling genomes with replacement.

    Args:
        genome_counts_dict: {test_accession: {'tp': int, 'fp': int, 'fn': int}} for one
            (read length, model) cell - i.e. per_genome_counts[length][model].
        n_boot: Number of bootstrap replicates.
        rng: numpy Generator (module-level, seeded, so results are reproducible).

    Returns:
        Dict of {metric: bootstrap std}, or NaN per metric if fewer than 2 genomes.
    """
    accessions = list(genome_counts_dict.keys())
    n = len(accessions)
    if n < 2:
        return {'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan}

    tp_arr = np.array([genome_counts_dict[a]['tp'] for a in accessions], dtype=float)
    fp_arr = np.array([genome_counts_dict[a]['fp'] for a in accessions], dtype=float)
    fn_arr = np.array([genome_counts_dict[a]['fn'] for a in accessions], dtype=float)

    idx = rng.integers(0, n, size=(n_boot, n))  # resample genome indices with replacement
    boot_scores = pooled_scores(tp_arr[idx].sum(axis=1),
                                fp_arr[idx].sum(axis=1),
                                fn_arr[idx].sum(axis=1))
    return {metric: float(values.std(ddof=1)) for metric, values in boot_scores.items()}
