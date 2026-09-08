"""
Postprocess the CAMI metagenome testset's CDS-labeled reads (output of
process_and_map_CAMI_testset.py) into the same testset_dict / testset_dict_30 /
read_names_list.pkl format that postprocess_testset.py produces for the
Mason-simulated genome testsets, so the metagenome results can be evaluated
by the same downstream pipeline.

Differs from postprocess_testset.py in data shape, not logic:
  - One TSV per CAMI sample (<sample>_cds_labels.tsv.gz) instead of one CSV
    per genome accession per error-model directory.
  - The indel/seq-error column is named "seq_errors" here (matching
    process_and_map_CAMI_testset.py's output), not "indel_positions".
  - cds_fragments_connection is always present (derived from real CIGARs, not
    simulated), so there's no with/without-indels split - it's always used.

EXCLUDE_UNCERTAIN_REGION_READS (below) optionally drops reads overlapping a
CDS_TAG_MARKERS-tagged annotation (hypothetical protein / ab initio prediction -
see process_and_map_CAMI_testset.py's uncertain_region_overlap column) from the
test set entirely, rather than just leaving them flagged. This is per-read: only
the individually-tagged mate is dropped, and its mate is kept (now an orphaned
single-end read) if it isn't itself tagged - each read's own annotation is judged
on its own merits regardless of its sibling's fate. Set it and re-run to produce a
second, stricter version of the test set under its own output directory - the
default (flag off) output is untouched either way.

EXCLUDE_NO_CDS_CONTIG_READS (below) optionally drops every read whose contig has ZERO
CDS features in its raw GFF3 at all (see test_CAMI/cami_contigs_without_cds.py, whose
output this reads back in). Distinct from EXCLUDE_UNCERTAIN_REGION_READS: that toggle is
about individual low-confidence annotations on an otherwise-annotated contig; this one is
about contigs NCBI hasn't annotated at all, where every read is a non-coding "negative" by
construction even though the contig may well carry real, unannotated genes - i.e. those
negatives are of unknown trustworthiness. Independent of and composable with
EXCLUDE_UNCERTAIN_REGION_READS (both may be on at once); requires
cami_contigs_without_cds.py to have already been run for the sample.
"""
import argparse
import ast
import glob
import os
import pickle
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# SCARB cluster data root - matches postprocess_testset.py and every other
# benchmark/postprocess script's project_root convention.
project_root = "/tmp/nrt204/FragmentPredictor"

data_dir = "CAMI_metagenome"  # matches predict_with_DeepCDS.py / predict_with_prodigal.py / predict_with_fgs.py

# When True, reads overlapping a hypothetical-protein / ab-initio-prediction-tagged CDS
# (uncertain_region_overlap=True in the source TSV) are dropped from the test set
# entirely - not counted as coding, and not counted as non-coding either, since the
# ground truth there is inherently less trustworthy than a curated annotation.
EXCLUDE_UNCERTAIN_REGION_READS = False

# See module docstring. Independent of EXCLUDE_UNCERTAIN_REGION_READS.
EXCLUDE_NO_CDS_CONTIG_READS = False


def load_no_cds_contigs(sample, project_root=project_root):
    """Contig accessions with zero CDS features in their raw GFF3 at all, persisted by
    test_CAMI/cami_contigs_without_cds.py. Only used when EXCLUDE_NO_CDS_CONTIG_READS is on."""
    contigs_path = Path(project_root) / "data" / "processed_data" / "CAMI_metagenome_no_cds_contigs" / f"{sample}_contigs_without_cds.txt"
    if not contigs_path.exists():
        raise FileNotFoundError(
            f"{contigs_path} not found - run test_CAMI/cami_contigs_without_cds.py --sample {sample} first."
        )
    return set(contigs_path.read_text().split())


def extract_indel_positions(seq_errors) -> list:
    """
    Extract positions that have insertions (I) or deletions (D) with their operations.

    Args:
        seq_errors (str): positions of insertion- or deletion error, e.g. "38I,73D"
    """
    if seq_errors == 'nan' or pd.isna(seq_errors):
        return []
    indel_pattern = r'(\d+[ID])'
    return re.findall(indel_pattern, seq_errors)


def get_errors_within_cds(cds_positions, sequencing_errors):
    """
    Get sequencing errors that fall within CDS boundaries.

    Args:
        cds_positions: List of CDS positions or any iterable of positions
        sequencing_errors: List of error strings like ['21D', '249I']

    Returns:
        List of error strings that fall within CDS boundaries
    """
    if not cds_positions or not sequencing_errors:
        return []

    min_pos = min(cds_positions)
    max_pos = max(cds_positions)

    errors_within_cds = []
    for error in sequencing_errors:
        position = int(error[:-1])  # Remove 'I' or 'D' and convert to int
        if min_pos <= position <= max_pos:
            errors_within_cds.append(error)

    return errors_within_cds


def process_metagenome_test_data(sample_tsv_path, project_root=project_root,
                                  exclude_uncertain_region_reads=EXCLUDE_UNCERTAIN_REGION_READS,
                                  exclude_no_cds_contig_reads=EXCLUDE_NO_CDS_CONTIG_READS) -> tuple:
    """
    Process a CAMI sample's CDS-labeled TSV (from process_and_map_CAMI_testset.py)
    into a structured dictionary format.

    Args:
        sample_tsv_path (str): Path to the sample's <sample>_cds_labels.tsv.gz file.
        exclude_uncertain_region_reads (bool): If True, reads flagged
            uncertain_region_overlap=True (overlapping a hypothetical-protein /
            ab-initio-prediction-tagged CDS) are dropped from the test set entirely,
            before either the coding-CDS dicts or all_test_read_names are built. This is
            per-read: a read's mate is unaffected unless it is itself tagged.
        exclude_no_cds_contig_reads (bool): If True, reads whose contig_accession has zero
            CDS features in its raw GFF3 at all (see load_no_cds_contigs /
            cami_contigs_without_cds.py) are dropped from the test set entirely. Independent
            of exclude_uncertain_region_reads.

    Returns:
        test_data_processed_dict (dict): A dictionary with CDS > 60bp.
        test_data_processed_dict_30 (dict): A dictionary with CDS > 30bp (superset of test_data_processed_dict).
        all_test_read_names (list): A list of all unique read names in the sample.
    """
    # Initialize
    test_data_processed_dict = dict()
    test_data_processed_dict_30 = dict()

    # Load sample's labeled reads
    test_data_df = pd.read_csv(sample_tsv_path, sep="\t", compression="gzip")

    if exclude_uncertain_region_reads:
        test_data_df = test_data_df[~test_data_df["uncertain_region_overlap"]]

    if exclude_no_cds_contig_reads:
        sample = os.path.basename(sample_tsv_path).replace("_cds_labels.tsv.gz", "")
        no_cds_contigs = load_no_cds_contigs(sample, project_root=project_root)
        test_data_df = test_data_df[~test_data_df["contig_accession"].isin(no_cds_contigs)]

    # Get all read names in test set
    all_test_read_names = list(set(list(test_data_df["read_name"])))

    # Process each read
    for _, row in test_data_df.iterrows():
        read_name = row["read_name"]
        cds_coords = ast.literal_eval(row["cds_coords"])
        cds_fragments_connection = ast.literal_eval(row["cds_fragments_connection"])

        seq_errors = row.get("seq_errors", None)
        indel_errors = extract_indel_positions(seq_errors) if seq_errors else []

        # If there is something coding in the read, store information
        if cds_coords != []:
            # Initialize inner dicts if not already present
            test_data_processed_dict[read_name] = dict()
            test_data_processed_dict[read_name]["cds_coords"] = []
            test_data_processed_dict[read_name]["cds_fragments_connection"] = []
            test_data_processed_dict[read_name]["seq_error_positions"] = []

            test_data_processed_dict_30[read_name] = dict()
            test_data_processed_dict_30[read_name]["cds_coords"] = []
            test_data_processed_dict_30[read_name]["cds_fragments_connection"] = []
            test_data_processed_dict_30[read_name]["seq_error_positions"] = []

            cds_connection_30_index = 0
            cds_connection_index = 0

            # Loop over each set of connected CDS coordinates
            for cds_connections in cds_fragments_connection:

                # Initialize
                cds_positions = []
                cds_fragments_to_store = []

                # Loop over each set of CDS fragments belonging to the same connected CDS
                for cds_frag_pos in cds_connections:
                    cds_coords_fragment = cds_coords[cds_frag_pos]

                    cds_fragments_to_store.append(cds_coords_fragment)
                    cds_positions += cds_coords_fragment[0:2]  # store start and stop coordinates

                # Figure out if CDS should go into "short fragments" (single-standing fragment shorter than 60 bp)
                full_cds_stretch = max(cds_positions) - min(cds_positions) + 1

                errors_in_cds = get_errors_within_cds(cds_positions, indel_errors)

                if full_cds_stretch > 60:
                    # Store in both testset_dict (>60bp) and testset_dict_30 (>30bp)
                    cds_connections_reindexed = [index_pos for index_pos in range(cds_connection_index, cds_connection_index + len(cds_connections))]
                    test_data_processed_dict[read_name]["cds_coords"] += cds_fragments_to_store
                    test_data_processed_dict[read_name]["cds_fragments_connection"].append(cds_connections_reindexed)
                    test_data_processed_dict[read_name]["seq_error_positions"] += errors_in_cds
                    cds_connection_index += len(cds_connections)

                    cds_connections_30_reindexed = [index_pos for index_pos in range(cds_connection_30_index, cds_connection_30_index + len(cds_connections))]
                    test_data_processed_dict_30[read_name]["cds_coords"] += cds_fragments_to_store
                    test_data_processed_dict_30[read_name]["cds_fragments_connection"].append(cds_connections_30_reindexed)
                    test_data_processed_dict_30[read_name]["seq_error_positions"] += errors_in_cds
                    cds_connection_30_index += len(cds_connections)

                elif full_cds_stretch >= 30 and full_cds_stretch <= 60:
                    # Store only in testset_dict_30 (>30bp)
                    cds_connections_30_reindexed = [index_pos for index_pos in range(cds_connection_30_index, cds_connection_30_index + len(cds_connections))]
                    test_data_processed_dict_30[read_name]["cds_coords"] += cds_fragments_to_store
                    test_data_processed_dict_30[read_name]["cds_fragments_connection"].append(cds_connections_30_reindexed)
                    test_data_processed_dict_30[read_name]["seq_error_positions"] += errors_in_cds
                    cds_connection_30_index += len(cds_connections)

    # Only include reads with coding sequences
    test_data_processed_dict = {
        read_name: data
        for read_name, data in test_data_processed_dict.items()
        if data["cds_coords"]  # Check CDS is not empty
    }

    test_data_processed_dict_30 = {
        read_name: data
        for read_name, data in test_data_processed_dict_30.items()
        if data["cds_coords"]  # Check CDS is not empty
    }

    return test_data_processed_dict, test_data_processed_dict_30, all_test_read_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-uncertain-region-reads", action="store_true", default=EXCLUDE_UNCERTAIN_REGION_READS,
                         help="Override EXCLUDE_UNCERTAIN_REGION_READS (module default: %(default)s)")
    parser.add_argument("--exclude-no-cds-contig-reads", action="store_true", default=EXCLUDE_NO_CDS_CONTIG_READS,
                         help="Override EXCLUDE_NO_CDS_CONTIG_READS (module default: %(default)s)")
    args = parser.parse_args()
    EXCLUDE_UNCERTAIN_REGION_READS = args.exclude_uncertain_region_reads
    EXCLUDE_NO_CDS_CONTIG_READS = args.exclude_no_cds_contig_reads

    input_dir = f"{project_root}/data/processed_data/reads_processed/test/{data_dir}"
    sample_tsv_paths = sorted(glob.glob(f"{input_dir}/*_cds_labels.tsv.gz"))
    if not sample_tsv_paths:
        raise FileNotFoundError(f"No *_cds_labels.tsv.gz files found in: {input_dir}")

    # Output-only directory name - the source TSVs are unaffected by the toggles, only
    # which reads make it into testset_processed/ here. Keeps all versions from
    # overwriting each other so they remain available for comparison.
    out_data_dir = (
        data_dir
        + ("_no_ab_initio" if EXCLUDE_UNCERTAIN_REGION_READS else "")
        + ("_no_cds_contigs" if EXCLUDE_NO_CDS_CONTIG_READS else "")
    )
    print(f"EXCLUDE_UNCERTAIN_REGION_READS={EXCLUDE_UNCERTAIN_REGION_READS}, "
          f"EXCLUDE_NO_CDS_CONTIG_READS={EXCLUDE_NO_CDS_CONTIG_READS} -> writing to testset_processed/{out_data_dir}/")

    for sample_tsv_path in tqdm(sample_tsv_paths, desc="Processing CAMI samples..."):
        sample = os.path.basename(sample_tsv_path).replace("_cds_labels.tsv.gz", "")
        print("Now processing sample: ", sample, flush=True)

        out_dir = f"{project_root}/data/processed_data/testset_processed/{out_data_dir}/{sample}"
        os.makedirs(out_dir, exist_ok=True)

        testset_dict, testset_dict_30, all_test_read_names_list = process_metagenome_test_data(
            sample_tsv_path,
            exclude_uncertain_region_reads=EXCLUDE_UNCERTAIN_REGION_READS,
            exclude_no_cds_contig_reads=EXCLUDE_NO_CDS_CONTIG_READS,
        )

        with open(f"{out_dir}/testset_dict.pkl", "wb") as processed_testset_file:
            pickle.dump(testset_dict, processed_testset_file)

        with open(f"{out_dir}/testset_dict_30.pkl", "wb") as processed_testset_30_file:
            pickle.dump(testset_dict_30, processed_testset_30_file)

        with open(f"{out_dir}/read_names_list.pkl", "wb") as read_names_file:
            pickle.dump(all_test_read_names_list, read_names_file)
