"""CDS matching functions of the CAMI benchmark, shared by the steps in this folder and by
../cami_hypothetical_protein_audit.
"""

def get_full_cds(reads_dict):
    """Merge fragmented CDS annotations that belong to the same gene.

    When a CDS is split into multiple fragments (e.g. due to short fragment
    removal during preprocessing), this function reconnects them into a single
    CDS span using the cds_fragments_connection mapping. The merged CDS is
    labelled 'disrupted_rf' and replaces the individual fragments.

    The original per-fragment coordinates (each with its own RF label) are
    preserved as a 4th element so that calculate_iou can determine which
    positions within the disrupted span belong to which reading frame:
        [start, end, 'disrupted_rf', [(s1, e1, rf1), (s2, e2, rf2), ...]]

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

                # Preserve individual fragment coordinates sorted by start position.
                fragment_list = sorted(
                    [(cds_coords[i][0], cds_coords[i][1], cds_coords[i][2])
                     for i in cds_frag_mapping],
                    key=lambda f: f[0]
                )

                indices_to_remove += cds_frag_mapping

                assert start_cds_frag < end_cds_frag, "Something went wrong processing indel-induced error in sequence!"
                cds_coords.append([start_cds_frag, end_cds_frag, 'disrupted_rf', fragment_list])
                reads_dict[read]['cds_coords'] = cds_coords

        for index in sorted(indices_to_remove, reverse=True):
            cds_coords.pop(index)

    return reads_dict


def _get_rf_spans(coord):
    """Extract per-RF coordinate spans from a CDS coord entry.

    Args:
        coord: [start, end, rf] or [start, end, 'disrupted_rf', [(s,e,rf), ...]].

    Returns:
        Dict mapping rf_label -> list of (start, end) tuples.
    """
    if len(coord) == 4 and coord[2] == 'disrupted_rf':
        by_rf = {}
        for s, e, rf in coord[3]:
            by_rf.setdefault(rf, []).append((s, e))
        return by_rf
    else:
        return {coord[2]: [(coord[0], coord[1])]}


def _overlap(s1, e1, s2, e2):
    return max(0, min(e1, e2) - max(s1, s2))


def _total_span_length(spans_by_rf):
    return sum(e - s for spans in spans_by_rf.values() for s, e in spans)


def _rf_intersection(spans_a, spans_b):
    """Total coordinate overlap between two lists of (start, end) spans (same RF)."""
    total = 0
    for s1, e1 in spans_a:
        for s2, e2 in spans_b:
            total += _overlap(s1, e1, s2, e2)
    return total


def calculate_iou(coord1, coord2):
    """Calculate IoU between two CDS entries, disrupted-RF-aware.

    coord1 is the prediction; coord2 is the truth.  Handles five cases:

    1. Both single-RF, same RF: standard coordinate IoU.
    2. Both single-RF, different RF: returns 0.0.
    3. Pred=single-RF, truth=disrupted_rf:
       IoU = intersection(pred, truth's matching-RF sub-span)
             / union(pred, truth's matching-RF sub-span).
       Truth's non-matching RF sub-spans are NOT counted as FN - the prediction
       only claims to have found one RF of the gene.
    4. Pred=disrupted_rf, truth=single-RF:
       TP = intersection(pred's matching-RF sub-span, truth).
       All non-matching RF sub-spans of pred count as FP (model predicted an
       indel that is absent in the truth).
       IoU = TP / (|pred_total| + |truth| - TP).
    5. Both disrupted_rf:
       RF-by-RF fragment comparison merged into one IoU:
       IoU = sum(intersection_rf) / (|pred_total| + |truth_total| - sum(intersection_rf)).

    Args:
        coord1: Prediction - [start, end, rf] or
                [start, end, 'disrupted_rf', [(s, e, rf), ...]].
        coord2: Truth - same format as coord1.

    Returns:
        IoU value in [0.0, 1.0].
    """
    pred_disrupted = coord1[2] == 'disrupted_rf'
    truth_disrupted = coord2[2] == 'disrupted_rf'

    # Cases 1 & 2: both single-RF
    if not pred_disrupted and not truth_disrupted:
        if coord1[2] != coord2[2]:
            return 0.0
        inter = _overlap(coord1[0], coord1[1], coord2[0], coord2[1])
        if inter == 0:
            return 0.0
        union = (coord1[1] - coord1[0]) + (coord2[1] - coord2[0]) - inter
        return inter / union if union > 0 else 0.0

    pred_spans = _get_rf_spans(coord1)
    truth_spans = _get_rf_spans(coord2)

    # Case 3: pred=single-RF, truth=disrupted_rf
    # Evaluate pred only against the matching-RF sub-span of the truth.
    if not pred_disrupted and truth_disrupted:
        rf = coord1[2]
        if rf not in truth_spans:
            return 0.0
        pred_len = coord1[1] - coord1[0]
        truth_rf_len = sum(e - s for s, e in truth_spans[rf])
        inter = _rf_intersection([(coord1[0], coord1[1])], truth_spans[rf])
        denom = pred_len + truth_rf_len - inter
        return inter / denom if denom > 0 else 0.0

    # Case 4: pred=disrupted_rf, truth=single-RF
    # Non-matching RF sub-spans of pred are FP.
    if pred_disrupted and not truth_disrupted:
        rf = coord2[2]
        truth_len = coord2[1] - coord2[0]
        pred_total_len = _total_span_length(pred_spans)
        inter = 0
        if rf in pred_spans:
            inter = _rf_intersection(pred_spans[rf], [(coord2[0], coord2[1])])
        denom = pred_total_len + truth_len - inter
        return inter / denom if denom > 0 else 0.0

    # Case 5: both disrupted_rf - RF-by-RF fragment comparison
    all_rfs = set(pred_spans.keys()) | set(truth_spans.keys())
    pred_total_len = _total_span_length(pred_spans)
    truth_total_len = _total_span_length(truth_spans)
    total_inter = 0
    for rf in all_rfs:
        if rf in pred_spans and rf in truth_spans:
            total_inter += _rf_intersection(pred_spans[rf], truth_spans[rf])
    denom = pred_total_len + truth_total_len - total_inter
    return total_inter / denom if denom > 0 else 0.0


def find_matching_genes(predicted_coords, actual_coords, overlap_criteria):
    """Match predicted CDS to ground-truth CDS using greedy IoU matching.

    Each predicted CDS is matched to the best-overlapping unmatched ground-truth
    CDS, provided the IoU meets the overlap_criteria threshold.

    Args:
        predicted_coords: List of CDS coord entries for predictions.
        actual_coords: List of CDS coord entries for ground-truth.
        overlap_criteria: Minimum IoU threshold for a valid match.

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

            if iou >= overlap_criteria and iou > best_iou:
                best_iou = iou
                best_actual_idx = actual_idx

        if best_actual_idx != -1:
            matched_pairs.append((pred_idx, best_actual_idx))
            used_actual.add(best_actual_idx)
            used_predicted.add(pred_idx)

    return matched_pairs, len(matched_pairs)


def calculate_metrics(testset_dict, model_preds_dict, all_test_read_names, overlap_criteria):
    """Calculate precision, recall, and F1 score at the CDS level.

    Iterates over all reads, matches predicted CDS to ground-truth CDS using
    IoU-based matching, and aggregates TP/FP/FN counts across all reads.

    Args:
        testset_dict: Dict mapping read names to ground-truth annotations with 'cds_coords'.
        model_preds_dict: Dict mapping read names to model predictions with 'cds_coords'.
        all_test_read_names: List of all read names to evaluate.
        overlap_criteria: Minimum IoU threshold for a match to count as a true positive.

    Returns:
        Dict with keys: overlap_criteria, total_predicted_genes, total_actual_genes,
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

        _, tp_count = find_matching_genes(cds_coords_model, cds_coords_testset, overlap_criteria)

        total_predicted_genes += num_predicted
        total_actual_genes += num_actual
        total_tp += tp_count

    total_fp = total_predicted_genes - total_tp
    total_fn = total_actual_genes - total_tp

    precision = total_tp / total_predicted_genes if total_predicted_genes > 0 else 0.0
    recall = total_tp / total_actual_genes if total_actual_genes > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    results = {
        'overlap_criteria': overlap_criteria,
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
