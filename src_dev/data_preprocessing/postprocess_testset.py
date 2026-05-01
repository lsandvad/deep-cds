import ast
import math
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

project_root = "../.."

test_accessions = open(f"{project_root}/data/processed_data/genome_partitions/test_partition_accessions.txt").read().splitlines()


def extract_indel_positions(seq_errors) -> list:
    """
    Extract positions that have insertions (I) or deletions (D) with their operations. 

    Args:
        seq_errors (str): positions of insertion- or deletion error, e.g. "38I,73D"
    """
    if seq_errors == 'nan' or pd.isna(seq_errors):
        return []
    # Find all patterns like "43D" or "210I" and capture both number and letter
    indel_pattern = r'(\d+[ID])'
    matches = re.findall(indel_pattern, seq_errors)
    # Return the matches as strings (e.g., ['132I', '144D'])
    return matches


def get_errors_within_cds(cds_positions, sequencing_errors):
    """
    Get sequencing errors that fall within CDS boundaries
    
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
        # Extract the position (everything except the last character)
        position = int(error[:-1])  # Remove 'I' or 'D' and convert to int
        
        # Check if position is within CDS boundaries
        if min_pos <= position <= max_pos:
            errors_within_cds.append(error)
    
    return errors_within_cds


def process_test_data(test_accession, testset_type, seq_len, indels_present, project_root=project_root) -> tuple:
    """
    Process test data from CSV file into a structured dictionary format.
    Args:
        test_accession (str): The accession identifier for the test dataset.
        testset_typw (str): The specific test set directory noting error rates and sequence lengths
        seq_len (int): length of sequences in specific test set 

    Returns:
        test_data_processed_dict (dict): A dictionary with CDS > 60bp.
        test_data_processed_dict_30 (dict): A dictionary with CDS > 30bp (superset of test_data_processed_dict).
        all_test_read_names (list): A list of all unique read names in the test dataset.
    """
    # Initialize
    test_data_processed_dict = dict()
    test_data_processed_dict_30 = dict()
    
    # Load test dataset
    test_data_df = pd.read_csv(f"{project_root}/data/processed_data/reads_processed/test/{testset_type}/csv/{test_accession}.csv.gz", index_col=0, compression="gzip")
    
    # Get all read names in test set 
    all_test_read_names = list(set(list(test_data_df["read_name"])))

    # Process each read
    for _, row in test_data_df.iterrows():
        read_name = row["read_name"]
        cds_coords = ast.literal_eval(row["cds_coords"])

        # If cds_fragments_connection exists, use it to connect indel-disrupted CDSs; otherwise treat each cds_coords as independent
        if indels_present:
            cds_fragments_connection = ast.literal_eval(row["cds_fragments_connection"])
        else:
            cds_fragments_connection = [[i] for i in range(len(cds_coords))]

        seq_errors = row.get("indel_positions", None)
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

                #Initialize
                cds_positions = []
                cds_fragments_to_store = []

                #Loop over each set of CDS fragments belonging to the same connected CDS
                for cds_frag_pos in cds_connections:
                    cds_coords_fragment = cds_coords[cds_frag_pos]

                    #if cds_coords_fragment[1] == seq_len:  #REMOVED
                    #    cds_coords_fragment[1] = seq_len - 3 

                    cds_fragments_to_store.append(cds_coords_fragment)
                    cds_positions += cds_coords_fragment[0:2] #store start and stop coordinates
                
                #Figure out if CDS should go into "short fragments" (single-standing fragment shorter than 60 bp)
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


    #Only include reads with coding sequences
    test_data_processed_dict = {
        read_name: data
        for read_name, data in test_data_processed_dict.items()
        if data["cds_coords"]  #Check CDS is not empty
    }

    test_data_processed_dict_30 = {
        read_name: data
        for read_name, data in test_data_processed_dict_30.items()
        if data["cds_coords"]  #Check CDS is not empty
    }

    return test_data_processed_dict, test_data_processed_dict_30, all_test_read_names


#Process test sets without sequencing errors
data_dir = ['without_errors_60bp', 
            'without_errors_75bp',  
            'without_errors_100bp', 
            'without_errors_150bp', 
            'without_errors_300bp', 
            'without_errors_700bp', 
            'without_errors_1000bp']
indels_present = False

for testset_type in data_dir:
    print("Now processing test set: ", testset_type, flush=True)
    seq_len = int(testset_type.split("_")[-1].strip("bp"))
        
    for test_accession in tqdm(test_accessions, desc="Processing test accessions..."):
        os.makedirs(f"{project_root}/data/processed_data/testset_processed/{testset_type}/{test_accession}", exist_ok=True)

        testset_dict, testset_dict_30, all_test_read_names_list = process_test_data(test_accession, testset_type, seq_len, indels_present=indels_present)

        with open(f"{project_root}/data/processed_data/testset_processed/{testset_type}/{test_accession}/testset_dict.pkl", "wb") as processed_testset_file:
            pickle.dump(testset_dict, processed_testset_file)

        with open(f"{project_root}/data/processed_data/testset_processed/{testset_type}/{test_accession}/testset_dict_30.pkl", "wb") as processed_testset_30_file:
            pickle.dump(testset_dict_30, processed_testset_30_file)

        with open(f"{project_root}/data/processed_data/testset_processed/{testset_type}/{test_accession}/read_names_list.pkl", "wb") as read_names_file:
            pickle.dump(all_test_read_names_list, read_names_file)


#Process testsets with sequencing errors 
data_dir = ['with_errors_5e-06i_0.004s_60bp',
 'with_errors_1.25e-05i_0.01s_60bp',
 'with_errors_3.75e-05i_0.03s_60bp',
 'with_errors_5e-06i_0.004s_75bp',
 'with_errors_1.25e-05i_0.01s_75bp',
 'with_errors_3.75e-05i_0.03s_75bp',
 'with_errors_5e-06i_0.004s_100bp',
 'with_errors_1.25e-05i_0.01s_100bp',
 'with_errors_3.75e-05i_0.03s_100bp',
 'with_errors_5e-06i_0.004s_150bp',
 'with_errors_1.25e-05i_0.01s_150bp',
 'with_errors_3.75e-05i_0.03s_150bp',
 'with_errors_5e-06i_0.004s_300bp',
 'with_errors_1.25e-05i_0.01s_300bp',
 'with_errors_3.75e-05i_0.03s_300bp',
 'HiSeq2500_150bp', 
 'MiSeq_v3_300bp', 
 'NextSeq500_150bp']

indels_present = True

for testset_type in data_dir:
    print("Now processing test set: ", testset_type, flush=True)
    seq_len = int(testset_type.split("_")[-1].strip("bp"))
        
    for test_accession in tqdm(test_accessions, desc="Processing test accessions..."):
        os.makedirs(f"{project_root}/data/processed_data/testset_processed/{testset_type}/{test_accession}", exist_ok=True)

        testset_dict, testset_dict_30, all_test_read_names_list = process_test_data(test_accession, testset_type, seq_len, indels_present=indels_present)

        with open(f"{project_root}/data/processed_data/testset_processed/{testset_type}/{test_accession}/testset_dict.pkl", "wb") as processed_testset_file:
            pickle.dump(testset_dict, processed_testset_file)

        with open(f"{project_root}/data/processed_data/testset_processed/{testset_type}/{test_accession}/testset_dict_30.pkl", "wb") as processed_testset_30_file:
            pickle.dump(testset_dict_30, processed_testset_30_file)

        with open(f"{project_root}/data/processed_data/testset_processed/{testset_type}/{test_accession}/read_names_list.pkl", "wb") as read_names_file:
            pickle.dump(all_test_read_names_list, read_names_file)