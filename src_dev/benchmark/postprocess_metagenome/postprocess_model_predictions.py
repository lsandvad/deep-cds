"""
Postprocess DeepCDS predictions on the CAMI metagenome test set (raw_predictions
from predict_metagenome/predict_with_DeepCDS.py) into the same model_preds_dict.pkl /
model_preds_dict_30.pkl format postprocess/postprocess_model_predictions.py produces
for the Mason-simulated genome testsets.

process_fragmented_cds and process_model_preds are unchanged from the original (the
enhanced-GFF format, including group_id-tagged frameshift fragments, is identical for
CAMI reads - and seq_len was never actually used by either function). What's different
is only the outer loop: instead of a hand-maintained matrix of model_type/testset_type/
model_preds_path combinations keyed by a fixed genome test_partition_accessions.txt,
this discovers whichever DeepCDS variant(s) (model_without_errors/with_substitution_errors/
with_errors) and CAMI sample(s) were actually predicted, by globbing
raw_predictions/DeepCDS/*/CAMI_metagenome/*/predictions_*.gff - so it processes exactly
what predict_with_DeepCDS.py produced, no more, no less.

EXCLUDE_UNCERTAIN_REGION_READS (below) mirrors postprocess_metagenome_testset.py's
toggle of the same name: when on, any read that postprocess_metagenome_testset.py
would drop from the test set (uncertain_region_overlap=True - overlapping a
hypothetical-protein / ab-initio-prediction-tagged CDS) is also dropped here from the
predictions, so a model isn't penalized as a false positive for calling a CDS on a
read whose ground truth we've decided not to trust.

EXCLUDE_NO_CDS_CONTIG_READS (below) mirrors postprocess_metagenome_testset.py's toggle
of the same name: when on, predictions on reads whose contig has zero CDS annotations
at all (see load_no_cds_contigs / test_CAMI/cami_contigs_without_cds.py) are dropped too.
Independent of and composable with EXCLUDE_UNCERTAIN_REGION_READS.
"""
import argparse
import glob
import os
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# SCARB cluster data root - matches postprocess/postprocess_model_predictions.py and every
# other benchmark/postprocess script's project_root convention.
project_root = "/tmp/nrt204/FragmentPredictor"

data_dir = "CAMI_metagenome"  # matches predict_with_prodigal.py / predict_with_fgs.py / predict_with_DeepCDS.py

# See postprocess_metagenome_testset.py's toggles of the same names.
EXCLUDE_UNCERTAIN_REGION_READS = False
EXCLUDE_NO_CDS_CONTIG_READS = False


def load_excluded_reads(sample, project_root=project_root):
    """read_names flagged uncertain_region_overlap=True in the sample's CDS-labeled TSV
    (hypothetical-protein / ab-initio-prediction-tagged ground truth). Only used when
    EXCLUDE_UNCERTAIN_REGION_READS is on. Per-read: a read's mate is unaffected unless
    it is itself tagged."""
    tsv_path = f"{project_root}/data/processed_data/reads_processed/test/{data_dir}/{sample}_cds_labels.tsv.gz"
    df = pd.read_csv(tsv_path, sep="\t", compression="gzip", usecols=["read_name", "uncertain_region_overlap"])
    return set(df.loc[df["uncertain_region_overlap"], "read_name"])


def load_no_cds_contigs(sample, project_root=project_root):
    """Contig accessions with zero CDS features in their raw GFF3 at all, persisted by
    test_CAMI/cami_contigs_without_cds.py."""
    contigs_path = Path(project_root) / "data" / "processed_data" / "CAMI_metagenome_no_cds_contigs" / f"{sample}_contigs_without_cds.txt"
    if not contigs_path.exists():
        raise FileNotFoundError(
            f"{contigs_path} not found - run test_CAMI/cami_contigs_without_cds.py --sample {sample} first."
        )
    return set(contigs_path.read_text().split())


def load_no_cds_contig_reads(sample, project_root=project_root):
    """read_names on a contig with zero CDS annotations at all (see load_no_cds_contigs).
    Only used when EXCLUDE_NO_CDS_CONTIG_READS is on."""
    no_cds_contigs = load_no_cds_contigs(sample, project_root=project_root)
    tsv_path = f"{project_root}/data/processed_data/reads_processed/test/{data_dir}/{sample}_cds_labels.tsv.gz"
    df = pd.read_csv(tsv_path, sep="\t", compression="gzip", usecols=["read_name", "contig_accession"])
    return set(df.loc[df["contig_accession"].isin(no_cds_contigs), "read_name"])


def process_fragmented_cds(model_preds_dict, model_dict_30):
    """
    Process fragmented CDS data to convert group labels, add error positions,
    and separate by CDS length. CDS > 60bp stays in main dict; CDS > 30bp goes into model_dict_30.
    """

    for read_name, data in model_preds_dict.items():
        cds_coords = data['cds_coords']
        cds_fragments_connection = data['cds_fragments_connection']

        # Step 1: Convert group labels to index-based connections
        new_connections = []
        group_dict = {}

        # First pass: collect all group members
        for i, connection in enumerate(cds_fragments_connection):
            if isinstance(connection, list) and len(connection) == 1:
                item = connection[0]
                if isinstance(item, str) and item.startswith('group_'):
                    if item not in group_dict:
                        group_dict[item] = []
                    group_dict[item].append(i)

        # Second pass: build new connections
        processed_indices = set()
        is_grouped = []
        for i, connection in enumerate(cds_fragments_connection):
            if i in processed_indices:
                continue

            if isinstance(connection, list) and len(connection) == 1:
                item = connection[0]
                if isinstance(item, str) and item.startswith('group_'):
                    # Add the group as a single connection
                    if item in group_dict:
                        new_connections.append(group_dict[item])
                        is_grouped.append(True)
                        processed_indices.update(group_dict[item])
                else:
                    # Regular single CDS
                    new_connections.append([item])
                    is_grouped.append(False)
                    processed_indices.add(i)

        # Step 2: Calculate CDS lengths and process errors
        valid_connections = []  # > 60bp, for main dict
        valid_connections_grouped = []  # > 60bp from groups only, for dict_30
        connections_30_only = []  # 30-60bp, only for dict_30
        seq_errors = []
        seq_errors_grouped = []
        errors_30_only = []

        for conn_idx, connection in enumerate(new_connections):
            # Calculate total CDS length for this connection
            if len(connection) == 1:
                # Single CDS
                cds = cds_coords[connection[0]]
                total_length = cds[1] - cds[0] + 1
                current_errors = []
            else:
                # Connected fragments - calculate total length
                fragments = [(i, cds_coords[i]) for i in connection]
                fragments.sort(key=lambda x: x[1][0])  # Sort by start position

                total_length = 0
                current_errors = []

                for j, (_, cds) in enumerate(fragments):
                    total_length += cds[1] - cds[0] + 1

                    # Calculate errors between fragments
                    if j < len(fragments) - 1:
                        current_cds = cds
                        next_cds = fragments[j + 1][1]

                        current_frame = int(current_cds[2])
                        next_frame = int(next_cds[2])

                        # Determine indel type by frame shift pattern
                        frame_shift = (next_frame - current_frame) % 3

                        if frame_shift == 1:
                            indel_type = 'I'
                        elif frame_shift == 2:
                            indel_type = 'D'
                        else:
                            continue

                        # Calculate gap midpoint
                        gap_start = current_cds[1] + 1
                        gap_end = next_cds[0] - 1
                        midpoint = (gap_start + gap_end) // 2

                        current_errors.append(f"{midpoint}{indel_type}")

            # Decide where to place this CDS based on length
            if total_length > 60:
                valid_connections.append(connection)
                seq_errors.extend(current_errors)
                # Only grouped CDS need to be added to dict_30 here;
                # non-grouped >60bp CDS are already in dict_30 from process_model_preds
                if is_grouped[conn_idx]:
                    valid_connections_grouped.append(connection)
                    seq_errors_grouped.extend(current_errors)

            elif total_length > 30:
                connections_30_only.append(connection)
                errors_30_only.extend(current_errors)

            # CDSs <= 30 bp are discarded

        # Update main dictionary (>60bp only)
        if valid_connections:
            # Rebuild cds_coords list with only valid CDSs and update indices
            new_cds_coords = []
            index_mapping = {}

            for connection in valid_connections:
                for old_idx in connection:
                    if old_idx not in index_mapping:
                        index_mapping[old_idx] = len(new_cds_coords)
                        new_cds_coords.append(cds_coords[old_idx])

            # Update connections with new indices
            final_connections = []
            for connection in valid_connections:
                final_connections.append([index_mapping[i] for i in connection])

            data['cds_coords'] = new_cds_coords
            data['cds_fragments_connection'] = final_connections
            data['seq_error_positions'] = seq_errors
        else:
            # No valid CDSs, mark for removal from main dict
            model_preds_dict[read_name] = None

        # Update dict_30 (>30bp) - only add grouped CDS (non-grouped already handled by process_model_preds)
        all_connections_30 = valid_connections_grouped + connections_30_only
        all_errors_30 = seq_errors_grouped + errors_30_only

        if all_connections_30:
            # Initialize if read doesn't exist in dict_30
            if read_name not in model_dict_30:
                model_dict_30[read_name] = {
                    'cds_coords': [],
                    'cds_fragments_connection': [],
                    'seq_error_positions': []
                }

            # Rebuild for dict_30
            new_30_coords = []
            index_mapping_30 = {}

            # Start indexing from existing cds_coords length
            existing_30_coords = model_dict_30[read_name]['cds_coords']
            start_idx = len(existing_30_coords)

            for connection in all_connections_30:
                for old_idx in connection:
                    if old_idx not in index_mapping_30:
                        index_mapping_30[old_idx] = start_idx + len(new_30_coords)
                        new_30_coords.append(cds_coords[old_idx])

            final_30_connections = []
            for connection in all_connections_30:
                final_30_connections.append([index_mapping_30[i] for i in connection])

            # Append to existing data
            model_dict_30[read_name]['cds_coords'].extend(new_30_coords)
            model_dict_30[read_name]['cds_fragments_connection'].extend(final_30_connections)
            model_dict_30[read_name]['seq_error_positions'].extend(all_errors_30)

    # Remove None entries from main dictionary
    model_preds_dict = {k: v for k, v in model_preds_dict.items() if v is not None and k is not None}

    return model_preds_dict, model_dict_30


def process_model_preds(sample, model_type, model_preds_path, excluded_reads=frozenset()):
    """
    Process model predictions from a GFF file for a given CAMI sample.

    Args:
        sample (str): The CAMI sample identifier, e.g. "sample_0".
        model_type (str): e.g. "DeepCDS/model_without_errors".
        model_preds_path (str): The model checkpoint subdirectory name.
        excluded_reads: read_names to drop from both results entirely (see
            EXCLUDE_UNCERTAIN_REGION_READS / load_excluded_reads).

    Returns:
        model_dict (dict): A dictionary with CDS > 60bp.
        model_dict_30 (dict): A dictionary with CDS > 30bp (superset of model_dict).
    """

    # Initialize
    model_dict = dict()
    model_dict_30 = dict()

    # Read model predictions GFF file
    with open(f"{project_root}/data/processed_data/predictions/raw_predictions/{model_type}/{data_dir}/{model_preds_path}/predictions_{sample}.gff", "r") as file:
        file.readline()  # Skip first line

        # Get CDS predictions for read
        for line in file:
            read_name = line.split("\t")[0]
            attr_desc = line.split("\t")[8]
            attr_type = line.split("\t")[2]

            if attr_type == "CDS":

                if read_name not in model_dict.keys():
                    model_dict[read_name] = dict()
                    model_dict[read_name]["cds_coords"] = []
                    model_dict[read_name]["cds_fragments_connection"] = []
                    model_dict[read_name]["seq_error_positions"] = []

                    model_dict_30[read_name] = dict()
                    model_dict_30[read_name]["cds_coords"] = []
                    model_dict_30[read_name]["cds_fragments_connection"] = []
                    model_dict_30[read_name]["seq_error_positions"] = []

                    counter_cds_on_read = 0
                    counter_cds_on_read_30 = 0

                # Get CDS coordinates and reading frame
                cds_start = int(line.split("\t")[3])
                cds_end = int(line.split("\t")[4])

                rf = str(int(line.split("\t")[7]))

                cds_coords = [cds_start, cds_end, rf]

                # If "group_id" is present, an indel has been predicted
                if "group_id" in attr_desc:
                    group_id = attr_desc.split("group_id=")[1].split(".")[0]
                    model_dict[read_name]["cds_coords"].append(cds_coords)
                    model_dict[read_name]["cds_fragments_connection"].append([group_id])

                else:
                    cds_length = cds_end - cds_start + 1
                    if cds_length > 60:
                        model_dict[read_name]["cds_coords"].append(cds_coords)
                        model_dict[read_name]["cds_fragments_connection"].append([counter_cds_on_read])
                        counter_cds_on_read += 1

                        model_dict_30[read_name]["cds_coords"].append(cds_coords)
                        model_dict_30[read_name]["cds_fragments_connection"].append([counter_cds_on_read_30])
                        counter_cds_on_read_30 += 1

                    elif cds_length >= 30:
                        model_dict_30[read_name]["cds_coords"].append(cds_coords)
                        model_dict_30[read_name]["cds_fragments_connection"].append([counter_cds_on_read_30])
                        counter_cds_on_read_30 += 1

    # Move short fragmented CDSs to model_dict_30 and add error positions to both dicts
    model_dict, model_dict_30 = process_fragmented_cds(model_dict, model_dict_30)

    # Reorganize to only include reads with coding sequences
    model_dict = {
        read_name: data
        for read_name, data in model_dict.items()
        if data["cds_coords"] and read_name not in excluded_reads
    }

    model_dict_30 = {
        read_name: data
        for read_name, data in model_dict_30.items()
        if data["cds_coords"] and read_name not in excluded_reads
    }

    return model_dict, model_dict_30


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-uncertain-region-reads", action="store_true", default=EXCLUDE_UNCERTAIN_REGION_READS,
                         help="Override EXCLUDE_UNCERTAIN_REGION_READS (module default: %(default)s)")
    parser.add_argument("--exclude-no-cds-contig-reads", action="store_true", default=EXCLUDE_NO_CDS_CONTIG_READS,
                         help="Override EXCLUDE_NO_CDS_CONTIG_READS (module default: %(default)s)")
    args = parser.parse_args()
    EXCLUDE_UNCERTAIN_REGION_READS = args.exclude_uncertain_region_reads
    EXCLUDE_NO_CDS_CONTIG_READS = args.exclude_no_cds_contig_reads

    gff_glob = f"{project_root}/data/processed_data/predictions/raw_predictions/DeepCDS/*/{data_dir}/*/predictions_*.gff"
    gff_paths = sorted(glob.glob(gff_glob))
    if not gff_paths:
        raise FileNotFoundError(f"No DeepCDS predictions found for {data_dir}: {gff_glob}")

    # Output-only directory name - raw predictions are unaffected by the toggles, only
    # which reads make it into processed_predictions/ here. Keeps all versions from
    # overwriting each other so they remain available for comparison.
    out_data_dir = (
        data_dir
        + ("_no_ab_initio" if EXCLUDE_UNCERTAIN_REGION_READS else "")
        + ("_no_cds_contigs" if EXCLUDE_NO_CDS_CONTIG_READS else "")
    )
    print(f"EXCLUDE_UNCERTAIN_REGION_READS={EXCLUDE_UNCERTAIN_REGION_READS}, "
          f"EXCLUDE_NO_CDS_CONTIG_READS={EXCLUDE_NO_CDS_CONTIG_READS} -> writing to processed_predictions/DeepCDS/*/{out_data_dir}/")

    _excluded_reads_by_sample = {}  # cache: several gff_paths (one per model variant) can share a sample

    for gff_path in tqdm(gff_paths, desc="Processing DeepCDS predictions for CAMI samples..."):
        p = Path(gff_path)
        sample = p.stem.replace("predictions_", "")
        model_preds_path = p.parent.name  # .../DeepCDS/{model_dir_path_suffix}/CAMI_metagenome/{model_preds_path}/predictions_{sample}.gff
        model_dir_path_suffix = p.parents[2].name
        model_type = f"DeepCDS/{model_dir_path_suffix}"

        print(f"{model_type} | {model_preds_path} | {sample}")

        if sample not in _excluded_reads_by_sample:
            excluded = set()
            if EXCLUDE_UNCERTAIN_REGION_READS:
                excluded |= load_excluded_reads(sample)
            if EXCLUDE_NO_CDS_CONTIG_READS:
                excluded |= load_no_cds_contig_reads(sample)
            _excluded_reads_by_sample[sample] = excluded
        excluded_reads = _excluded_reads_by_sample.get(sample, frozenset())

        out_dir = f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{out_data_dir}/{model_preds_path}/{sample}"
        os.makedirs(out_dir, exist_ok=True)

        model_preds_dict, model_preds_dict_30 = process_model_preds(
            sample, model_type, model_preds_path, excluded_reads=excluded_reads
        )

        with open(f"{out_dir}/model_preds_dict.pkl", "wb") as processed_preds_file:
            pickle.dump(model_preds_dict, processed_preds_file)

        with open(f"{out_dir}/model_preds_dict_30.pkl", "wb") as processed_preds_file:
            pickle.dump(model_preds_dict_30, processed_preds_file)
