import ast
import math
import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

project_root = "../../.." #"/tmp/nrt204/FragmentPredictor" #local "../.."../

test_accessions = open(f"{project_root}/data/processed_data/genome_partitions/test_partition_accessions.txt").read().splitlines()

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


def process_model_preds(test_accession, model_type, testset_type, model_preds_path, seq_len):
    """ 
    Process model predictions from a GFF file for a given test accession.

    Args:
        test_accession (str): The accession identifier for the test dataset.
        model_preds_path (str): The path to the model predictions directory.

    Returns:
        model_dict (dict): A dictionary with CDS > 60bp.
        model_dict_30 (dict): A dictionary with CDS > 30bp (superset of model_dict).
    """

    #Initialize
    model_dict = dict()
    model_dict_30 = dict()

    #Read model predictions GFF file
    with open(f"{project_root}/data/processed_data/predictions/raw_predictions/{model_type}/{testset_type}/{model_preds_path}/predictions_{test_accession}.gff", "r") as file:
        file.readline() #Skip first line

        #Get CDS predictions for read
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

                #Get CDS coordinates and reading frame
                cds_start = int(line.split("\t")[3])
                cds_end = int(line.split("\t")[4])
                
                rf = str(int(line.split("\t")[7]))

                cds_coords = [cds_start, cds_end, rf]

                #If "group_id" is present, an indel has been predicted
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

    #Move short fragmented CDSs to model_dict_30 and add error positions to both dicts
    model_dict, model_dict_30 = process_fragmented_cds(model_dict, model_dict_30)

    #Reorganize to only include reads with coding sequences
    model_dict = {
        read_name: data 
        for read_name, data in model_dict.items() 
        if data["cds_coords"]  #Check if cds_coords is not empty
    }

    model_dict_30 = {
        read_name: data 
        for read_name, data in model_dict_30.items() 
        if data["cds_coords"]  #Check if cds_coords is not empty
    }

    return model_dict, model_dict_30


##########################################################################################################################################
###############Process preds for data without sequencing errors (DeepCDS (Full), smaller datasets for model training)#####################
##########################################################################################################################################

#Navigate to DeepCDS without sequencing errors predictions directory
model_type = "DeepCDS/model_without_errors"

#Process preds for model trained on 100, 200, and 400 genomes (only 300bp dataset)
testset_types = ["without_errors_300bp"]
model_preds_paths = ["full_model_100_genomes_seed_42_trained_final_8M_no_dropout", 
                     "full_model_200_genomes_seed_42_trained_final_8M_no_dropout", 
                     "full_model_400_genomes_seed_42_trained_final_8M_no_dropout"]

for testset_type in testset_types:
    print(testset_type)

    seq_len = int(testset_type.split("_")[-1].strip("bp"))

    for model_preds_path in tqdm(model_preds_paths, desc="Processing predictions for model type..."):
        for test_accession in test_accessions:
            os.makedirs(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}", exist_ok=True)
            
            model_preds_dict, model_preds_dict_30 = process_model_preds(test_accession, model_type, testset_type, model_preds_path, seq_len)

            with open(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}/model_preds_dict.pkl", "wb") as processed_preds_file:
                pickle.dump(model_preds_dict, processed_preds_file)

            with open(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}/model_preds_dict_30.pkl", "wb") as processed_preds_file:
                pickle.dump(model_preds_dict_30, processed_preds_file)





##########################################################################################################################################
##################################Process preds for data without sequencing errors (DeepCDS (Full))#######################################
##########################################################################################################################################

#Navigate to DeepCDS without sequencing errors predictions directory
model_type = "DeepCDS/model_without_errors"
model_preds_paths = ["full_model_all_genomes_seed_42_trained_final_8M_no_dropout"]

#Process preds for model trained on all genomes 
testset_types = ["without_errors_60bp",
                "without_errors_75bp",
                "without_errors_100bp",
                "without_errors_150bp",
                "without_errors_300bp",
                "without_errors_700bp",
                "without_errors_1000bp"]


for testset_type in testset_types:
    print(testset_type)

    seq_len = int(testset_type.split("_")[-1].strip("bp"))

    for model_preds_path in tqdm(model_preds_paths, desc="Processing predictions for model type..."):
        for test_accession in test_accessions:
            os.makedirs(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}", exist_ok=True)
            
            model_preds_dict, model_preds_dict_30 = process_model_preds(test_accession, model_type, testset_type, model_preds_path, seq_len)

            with open(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}/model_preds_dict.pkl", "wb") as processed_preds_file:
                pickle.dump(model_preds_dict, processed_preds_file)

            with open(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}/model_preds_dict_30.pkl", "wb") as processed_preds_file:
                pickle.dump(model_preds_dict_30, processed_preds_file)



##########################################################################################################################################
##################################Process preds for data without sequencing errors (DeepCDS (pLM))########################################
##########################################################################################################################################

#Navigate to DeepCDS without sequencing errors predictions directory
model_type = "DeepCDS_A1/model_without_errors"
model_preds_paths = ["esm2_8m_all_genomes_seed_42_trained_final_8M_no_dropout"]

#Process preds for model trained on all genomes 
testset_types = ["without_errors_60bp",
                "without_errors_75bp",
                "without_errors_100bp",
                "without_errors_150bp",
                "without_errors_300bp",
                "without_errors_700bp",
                "without_errors_1000bp"]


for testset_type in testset_types:
    print(testset_type)

    seq_len = int(testset_type.split("_")[-1].strip("bp"))

    for model_preds_path in tqdm(model_preds_paths, desc="Processing predictions for model type..."):
        for test_accession in test_accessions:
            os.makedirs(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}", exist_ok=True)
            
            model_preds_dict, model_preds_dict_30 = process_model_preds(test_accession, model_type, testset_type, model_preds_path, seq_len)

            with open(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}/model_preds_dict.pkl", "wb") as processed_preds_file:
                pickle.dump(model_preds_dict, processed_preds_file)

            with open(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}/model_preds_dict_30.pkl", "wb") as processed_preds_file:
                pickle.dump(model_preds_dict_30, processed_preds_file)



##########################################################################################################################################
###############Process preds for data with sequencing errors (DeepCDS S+I (Full), smaller datasets for model training)####################
##########################################################################################################################################

model_types = ["DeepCDS/model_with_errors"]

#Process preds for model trained on 100, 200, and 400 genomes (only 300bp dataset)
testset_types = ["with_errors_1.25e-05i_0.01s_300bp",
                 "with_errors_5e-06i_0.004s_300bp",
                 "with_errors_3.75e-05i_0.03s_300bp"]

model_preds_paths = ["full_model_100_genomes_seed_42_trained_final_8M_no_dropout", 
                     "full_model_200_genomes_seed_42_trained_final_8M_no_dropout", 
                     "full_model_400_genomes_seed_42_trained_final_8M_no_dropout"]

for model_type in model_types:
    for testset_type in testset_types:
        print(testset_type)

        seq_len = int(testset_type.split("_")[-1].strip("bp"))

        for model_preds_path in tqdm(model_preds_paths, desc="Processing predictions for model type..."):
            try:
                for test_accession in test_accessions:
                    os.makedirs(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}", exist_ok=True)
                    
                    model_preds_dict, model_preds_dict_30 = process_model_preds(test_accession, model_type, testset_type, model_preds_path, seq_len)

                    with open(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}/model_preds_dict.pkl", "wb") as processed_preds_file:
                        pickle.dump(model_preds_dict, processed_preds_file)

                    with open(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}/model_preds_dict_30.pkl", "wb") as processed_preds_file:
                        pickle.dump(model_preds_dict_30, processed_preds_file)
            except FileNotFoundError as err:
                print(err)
                continue


##########################################################################################################################################
##################Process preds for data with sequencing errors (DeepCDS (Full), DeepCDS S (Full), DeepCDS S+I (Full)#####################
##########################################################################################################################################

model_types = ["DeepCDS/model_with_errors", 
               "DeepCDS/model_with_substitution_errors", 
               "DeepCDS/model_without_errors"]

testset_types = ["with_errors_1.25e-05i_0.01s_60bp",
                 "with_errors_1.25e-05i_0.01s_75bp",
                 "with_errors_1.25e-05i_0.01s_100bp",
                 "with_errors_1.25e-05i_0.01s_150bp",
                 "with_errors_1.25e-05i_0.01s_300bp",
                 "with_errors_5e-06i_0.004s_60bp",
                 "with_errors_5e-06i_0.004s_75bp",
                 "with_errors_5e-06i_0.004s_100bp",
                 "with_errors_5e-06i_0.004s_150bp",
                 "with_errors_5e-06i_0.004s_300bp",
                 "with_errors_3.75e-05i_0.03s_60bp",
                 "with_errors_3.75e-05i_0.03s_75bp",
                 "with_errors_3.75e-05i_0.03s_100bp",
                 "with_errors_3.75e-05i_0.03s_150bp",
                 "with_errors_3.75e-05i_0.03s_300bp",
                 "HiSeq2500_150bp",     #Simulated with modern_art
                 "MiSeq_v3_300bp",      #Simulated with modern_art 
                 "NextSeq500_150bp"]    #Simulated with modern_art

model_preds_paths = ["full_model_all_genomes_seed_42_trained_final_8M_no_dropout"]

for testset_type in testset_types:
    print(testset_type)
    
    seq_len = int(testset_type.split("_")[-1].strip("bp"))

    for model_type in model_types:
        
        for model_preds_path in model_preds_paths:
            try:
                for test_accession in test_accessions:
                    os.makedirs(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}", exist_ok=True)
                    
                    model_preds_dict, model_preds_dict_30 = process_model_preds(test_accession, model_type, testset_type, model_preds_path, seq_len)

                    with open(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}/model_preds_dict.pkl", "wb") as processed_preds_file:
                        pickle.dump(model_preds_dict, processed_preds_file)

                    with open(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}/model_preds_dict_30.pkl", "wb") as processed_preds_file:
                        pickle.dump(model_preds_dict_30, processed_preds_file)
            
            except FileNotFoundError:
                print("Not found!")
                continue



##########################################################################################################################################
####################Process preds for data with sequencing errors (DeepCDS (pLM), DeepCDS S (pLM), DeepCDS S+I (pLM#)#####################
##########################################################################################################################################
model_types = ["DeepCDS_A1/model_with_errors", 
               "DeepCDS_A1/model_with_substitution_errors", 
               "DeepCDS_A1/model_without_errors"]

testset_types = ["with_errors_1.25e-05i_0.01s_60bp",
                 "with_errors_1.25e-05i_0.01s_75bp",
                 "with_errors_1.25e-05i_0.01s_100bp",
                 "with_errors_1.25e-05i_0.01s_150bp",
                 "with_errors_1.25e-05i_0.01s_300bp",
                 "with_errors_5e-06i_0.004s_60bp",
                 "with_errors_5e-06i_0.004s_75bp",
                 "with_errors_5e-06i_0.004s_100bp",
                 "with_errors_5e-06i_0.004s_150bp",
                 "with_errors_5e-06i_0.004s_300bp",
                 "with_errors_3.75e-05i_0.03s_60bp",
                 "with_errors_3.75e-05i_0.03s_75bp",
                 "with_errors_3.75e-05i_0.03s_100bp",
                 "with_errors_3.75e-05i_0.03s_150bp",
                 "with_errors_3.75e-05i_0.03s_300bp",
                 "HiSeq2500_150bp",         #Simulated with modern_art
                 "MiSeq_v3_300bp",          #Simulated with modern_art
                 "NextSeq500_150bp"]        #Simulated with modern_art

model_preds_paths = ["esm2_8m_all_genomes_seed_42_trained_final_8M_no_dropout"]

for testset_type in testset_types:
    print(testset_type)
    
    seq_len = int(testset_type.split("_")[-1].strip("bp"))

    for model_type in model_types:
        
        for model_preds_path in model_preds_paths:
            try:
                for test_accession in test_accessions:
                    os.makedirs(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}", exist_ok=True)
                    
                    model_preds_dict, model_preds_dict_30 = process_model_preds(test_accession, model_type, testset_type, model_preds_path, seq_len)

                    with open(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}/model_preds_dict.pkl", "wb") as processed_preds_file:
                        pickle.dump(model_preds_dict, processed_preds_file)

                    with open(f"{project_root}/data/processed_data/predictions/processed_predictions/{model_type}/{testset_type}/{model_preds_path}/{test_accession}/model_preds_dict_30.pkl", "wb") as processed_preds_file:
                        pickle.dump(model_preds_dict_30, processed_preds_file)
            
            except FileNotFoundError:
                print("Not found!")
                continue
