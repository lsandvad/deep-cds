import gzip
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import pysam
from Bio import SeqIO

####NEXT STEP: USE RUFF, FORMAT, REMOVE .ipynb from github, ADD .py to github, update readme!!

# Estimate combination sets from data
# Purpose: Map label class IDs to 3D encodings (label class IDs for rf0, rf1, rf2)
# Advantage: ignores all combinations not occuring in data, which can for this reason not be learned from


def get_rf_combinations(rf0_labels, rf1_labels, rf2_labels) -> set:
    """
    Extract all unique combinations of labels at corresponding positions
    across three reading frames.

    Args:
        rf0_labels, rf1_labels, rf2_labels: Lists of labels for each reading frame

    Returns:
        set: All unique (rf0, rf1, rf2) combinations found in the data
    """

    rf0_labels = rf0_labels[:-1]

    # Ensure all reading frames have the same length
    assert len(rf0_labels) == len(rf1_labels) == len(rf2_labels), "All reading frames must have the same length"

    combinations = set()

    # Iterate through all positions
    for i in range(len(rf0_labels)):
        combination = (rf0_labels[i], rf1_labels[i], rf2_labels[i])
        combinations.add(combination)

    return combinations


def collect_label_transition_combinations(archive, error_rates, all_combinations, accession_filenames) -> set:
    """
    Collect all unique label combinations across multiple processed read files.

    Args:
        archive (str): Data partiton
        error_rates (str): Error rate category (e.g., "model_with_errors")
        all_combinations (set): Set to store all unique label combinations, update iteratively
        accession_filenames (list): List of accession filenames to process
    """

    # for accession_filename in tqdm(accession_filenames):
    for accession_filename in accession_filenames:
        processed_reads = pd.read_csv(f"../../data/processed_data/reads_processed/{archive}/{error_rates}/csv/{accession_filename}.csv.gz", compression="gzip", index_col=0, low_memory=False)

        # Process all rows at once
        for _, row in processed_reads.iterrows():
            # Dictionary to store labels per reading frame
            labels = {}

            # Make each RF into separate samples
            for frame in ["rf0", "rf1", "rf2"]:
                # Format labels from string to list of integers
                frame_labels = row[f"{frame}_labels"].replace(" ", "").replace("[", "").replace("]", "").split(",")
                labels[frame] = [int(label) for label in frame_labels]

                # Label any indel transitions (instead of going between 0 and 1, they go from 0 -> 4 -> 1 or 1 -> 5 -> 0)
                if 1 in labels[frame] and 0 in labels[frame]:
                    # Find all transitions from 0 to 1 (mark as 4)
                    for i in range(1, len(labels[frame])):
                        if labels[frame][i] == 1 and labels[frame][i - 1] == 0:
                            labels[frame][i] = 4

                    # Find all transitions from 1 to 0 (mark the last 1 in each stretch as 5)
                    for i in range(len(labels[frame]) - 1):
                        if labels[frame][i] == 1 and labels[frame][i + 1] == 0:
                            labels[frame][i] = 5

            sample_combinations = get_rf_combinations(labels["rf0"], labels["rf1"], labels["rf2"])
            all_combinations.update(sample_combinations)

    return all_combinations


if __name__ == "__main__":
    train_accessions = open("../../data/processed_data/genome_partitions/train_partition_accessions.txt").read().splitlines()
    val_accessions = open("../../data/processed_data/genome_partitions/val_partition_accessions.txt").read().splitlines()

    os.makedirs("../../data/processed_data/model_data/shared_crf/model_with_errors/label_mappings/", exist_ok=True)
    os.makedirs("../../data/processed_data/model_data/shared_crf/model_without_errors/label_mappings/", exist_ok=True)
    os.makedirs("../../data/processed_data/model_data/shared_crf/model_with_substitution_errors/label_mappings/", exist_ok=True)

    # Get label encodings for data with sequencing errors
    all_combinations = set()
    all_combinations = collect_label_transition_combinations("train_val", "with_errors", all_combinations, train_accessions + val_accessions)

    mapping_dict_to_3d_vector = dict()
    mapping_dict_to_class = dict()
    all_combinations = sorted(all_combinations)

    for i in range(len(all_combinations)):
        mapping_dict_to_3d_vector[i] = all_combinations[i]
        mapping_dict_to_class[all_combinations[i]] = i

    with open("../../data/processed_data/model_data/shared_crf/model_with_errors/label_mappings/mapping_to_3d_vector.pkl", "wb") as mapping_file:
        pickle.dump(mapping_dict_to_3d_vector, mapping_file)

    with open("../../data/processed_data/model_data/shared_crf/model_with_errors/label_mappings/mapping_to_class.pkl", "wb") as mapping_file:
        pickle.dump(mapping_dict_to_class, mapping_file)

    # Get label encodings for data with substitution errors only
    all_combinations = set()
    all_combinations = collect_label_transition_combinations("train_val", "with_substitution_errors", all_combinations, train_accessions + val_accessions)

    mapping_dict_to_3d_vector = dict()
    mapping_dict_to_class = dict()
    all_combinations = sorted(all_combinations)

    for i in range(len(all_combinations)):
        mapping_dict_to_3d_vector[i] = all_combinations[i]
        mapping_dict_to_class[all_combinations[i]] = i

    with open("../../data/processed_data/model_data/shared_crf/model_with_substitution_errors/label_mappings/mapping_to_3d_vector.pkl", "wb") as mapping_file:
        pickle.dump(mapping_dict_to_3d_vector, mapping_file)

    with open("../../data/processed_data/model_data/shared_crf/model_with_substitution_errors/label_mappings/mapping_to_class.pkl", "wb") as mapping_file:
        pickle.dump(mapping_dict_to_class, mapping_file)

    # Get label encodings for data without sequencing errors at all
    all_combinations = set()
    all_combinations = collect_label_transition_combinations("train_val", "without_errors", all_combinations, train_accessions + val_accessions)

    mapping_dict_to_3d_vector = dict()
    mapping_dict_to_class = dict()
    all_combinations = sorted(all_combinations)

    for i in range(len(all_combinations)):
        mapping_dict_to_3d_vector[i] = all_combinations[i]
        mapping_dict_to_class[all_combinations[i]] = i

    with open("../../data/processed_data/model_data/shared_crf/model_without_errors/label_mappings/mapping_to_3d_vector.pkl", "wb") as mapping_file:
        pickle.dump(mapping_dict_to_3d_vector, mapping_file)

    with open("../../data/processed_data/model_data/shared_crf/model_without_errors/label_mappings/mapping_to_class.pkl", "wb") as mapping_file:
        pickle.dump(mapping_dict_to_class, mapping_file)
