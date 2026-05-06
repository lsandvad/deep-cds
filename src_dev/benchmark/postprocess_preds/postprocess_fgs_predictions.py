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

project_root = "/tmp/nrt204/FragmentPredictor" #local "../.."

test_accessions = open(f"{project_root}/data/processed_data/genome_partitions/test_partition_accessions.txt").read().splitlines()

def trim_cds(start, stop, frame):
    """
    Trim CDS coordinates [start, stop] (1-indexed) to include
    only complete codons in the correct reading frame.

    Args:
        start (int): Start coordinate (1-indexed)
        stop (int): Stop coordinate (end position in last coding codon; 1-indexed)
        frame (int): Reading frame (0, 1, or 2)

    Returns: 
        Trimmed inputs (start, stop, frame)
    """

    #Adjust start forward to first valid codon in frame
    while (start - frame) % 3 != 1:
        start += 1

    #Adjust stop backward to preserve full codons
    while (stop - start + 1) % 3 != 0:
        stop -= 1

    return start, stop, frame


def get_cds_chunks_on_read(cds_start_read, cds_end_read, insertions, deletions, initial_rf):
    """
    Map back the chunks of the CDS to the original read.
    When an indel has been predicted, FGS returns the full CDS with the initital reading frame and with CDS coordinates on the read it was derived from. 
    instead of the fragmented CDS coordinates within different reading frames, which is why this postprocessing is needed. 
    Indel positions are excluded from the fragmented CDS chunks.

    Args:
        cds_start_read (int): The CDS start position on the read.
        cds_end_read (int): The CDS stop position on the read.
        insertions: positions that FGS predicts a nucleotide was inserted on the read
        deletions: positions that FGS predicts a nucleotide was deleted on the read
        initial_rf (0, 1, 2): The reading frame that the CDS begins in before any indel errors.

    Returns:
        corrected_complete_cds_coords (list of lists): The corrected, complete CDS fragments belonging to the same overall CDS

    """

    #Combine all indel positions and sort them (the points where we must break the sequence)
    break_points = sorted(set(insertions + deletions))
    
    #Initialize
    chunks = []
    current_start = cds_start_read
    current_rf = initial_rf
    corrected_complete_cds_coords = []
    
    #Process all break points plus the final end point
    all_breaks = break_points + [cds_end_read + 1]  # +1 to include the end in the last loop

    #Iterate through each break point
    for break_pos in all_breaks:
        #The end of the current contiguous chunk is the position BEFORE the break
        chunk_end = break_pos - 1
        
        #Only create a chunk if the current start is before or equal to the calculated end
        if current_start <= chunk_end and current_start <= cds_end_read and chunk_end >= cds_start_read:
            #Cut the chunk to the CDS boundaries
            effective_start = max(current_start, cds_start_read)
            effective_end = min(chunk_end, cds_end_read)
            if effective_start <= effective_end:
                chunks.append([effective_start, effective_end, current_rf])
        
        #update the reading frame for the next CDS chunk based on what type of break this is
        if break_pos in deletions:
            #A deletion means we are MISSING a base at break_pos in the read -> frame pushed forward by 1
            current_rf = (current_rf - 1) % 3
            #The next chunk starts at the break_pos itself;
            #The base that should be at break_pos is missing, so the next available base is at break_pos.
            current_start = break_pos
            
        elif break_pos in insertions:
            #An insertion means we have an EXTRA base at break_pos in the read;
            #We skip it, so the next base in the read is at break_pos + 1 -> frame psuhed back by 1
            current_rf = (current_rf + 1) % 3
            #The next chunk starts after the inserted base.
            current_start = break_pos + 1
        
        #end of CDS
        else:
            current_start = break_pos

    #Trim fragmented CDS coords to include only complete codons
    for cds_coords_fragment in chunks:
        cds_start = cds_coords_fragment[0]
        cds_stop = cds_coords_fragment[1]
        rf = cds_coords_fragment[2]

        cds_start, cds_stop, rf = trim_cds(cds_start, cds_stop, rf)

        #Add CDS fragments except fragmented predicted codons caused by predicted indels placed in close proximity
        if (abs(cds_stop - cds_start) + 1) % 3 == 0:
            corrected_complete_cds_coords.append([cds_start, cds_stop, str(rf)])
        else:
            assert cds_stop - cds_start <= 1 #Ensure that only disrupted codons (spanning maximum 2 nucleotide positions) are removed

    return corrected_complete_cds_coords



def remap_and_process_fgs_preds(test_accession, error_model, seq_len):
    """ 
    Remap predicted CDS coordinates to match format of test set for benchmark.
    """ 

    #Load predicted fragments with indel error
    preds_info_path = f"{project_root}/data/processed_data/predictions/raw_predictions/fgs_preds/{error_model}/{test_accession}/{test_accession}.out"

    #Initialize
    fgs_preds_dict = dict()

    #Iterate over each read prediction information in the preds_info file
    with open(preds_info_path, 'r') as preds_info_file:
        for line in preds_info_file:

            #Entry line for read
            if line.startswith('>'):
                #Extract read ID
                read_id = line[1:].strip().split("|")[0]
                fgs_preds_dict[read_id] = dict()
                fgs_preds_dict[read_id]["cds_coords"] = []
                fgs_preds_dict[read_id]["cds_fragments_connection"] = []
                fgs_preds_dict[read_id]["seq_error_positions"] = []
                cds_fragments_connections_counter = 0

            #Prediction lines for read (each line contains predictions for one CDS; there can be mutiple on one read)
            else:
                cds_coord_info = line.strip().split("\t")
                #All reads have been processed to "template" strand; remove those detected on complement strand 
                if cds_coord_info[2] == "+":
                    #Get start and stop coordinate for CDS
                    start_coord = int(cds_coord_info[0])
                    end_coord = int(cds_coord_info[1])

                    #ADDED DUE TO BUG IN FGS
                    if end_coord == seq_len - 3:
                        end_coord = seq_len

                    #rf in fgs prediction files are (1, 2, 3); change to (0, 1, 2)
                    rf = int(cds_coord_info[3]) - 1

                    #get all predicted insertions and deletions involved with predicted CDS and turn into lists
                    insertions = cds_coord_info[5].strip("I:").split(",")
                    deletions = cds_coord_info[6].strip("D:").split(",")
                    insertions_list = [int(insertion_pos) for insertion_pos in insertions if insertion_pos != ""]
                    deletions_list = [int(deletion_pos) for deletion_pos in deletions if deletion_pos != ""]

                    ##In no insertions and deletions are predicted in CDS, just return predicted coords
                    if insertions_list == [] and deletions_list == []:
                        coord_info = [start_coord, end_coord, str(rf)]
                        fgs_preds_dict[read_id]["cds_coords"].append(coord_info)
                        fgs_preds_dict[read_id]["cds_fragments_connection"].append([cds_fragments_connections_counter])
                        cds_fragments_connections_counter += 1

                    #Run over reads containing indels
                    else:
                        #Get CDS fragments interrupted due to indels
                        mapped_cds_fragments_on_read = get_cds_chunks_on_read(start_coord, end_coord, insertions_list, deletions_list, rf)
                        fgs_preds_dict[read_id]["cds_coords"] += mapped_cds_fragments_on_read

                        #Store information of which disrupted fragments belong "together" 
                        cds_frags = len(mapped_cds_fragments_on_read)
                        cds_frags_pos = []

                        for i in range(cds_frags):
                            cds_frags_pos.append(cds_fragments_connections_counter)
                            cds_fragments_connections_counter += 1

                        fgs_preds_dict[read_id]["cds_fragments_connection"].append(cds_frags_pos)

                        #Store predicted indel positions
                        indels = [str(insertion_pos)+"I" for insertion_pos in insertions_list] + [str(deletion_pos)+"D" for deletion_pos in deletions_list]
                        fgs_preds_dict[read_id]["seq_error_positions"] += indels
    
    #Filter out all read ids with CDS predicted only on complement strand
    fgs_preds_dict = {
        read_id: data 
        for read_id, data in fgs_preds_dict.items() 
        if data["cds_coords"] != []}

    return fgs_preds_dict
                


#Process preds without sequencing erros 
pred_dirs = os.listdir(f"{project_root}/data/processed_data/predictions/raw_predictions/fgs_preds/")
pred_dirs = [dir for dir in pred_dirs if dir != ".DS_Store"]
pred_dirs = [dir for dir in pred_dirs if dir != "archive"]

for pred_data in tqdm(pred_dirs):
    os.makedirs(f"{project_root}/data/processed_data/predictions/processed_predictions/fgs_preds/{pred_data}", exist_ok=True)
    print(pred_data)
    seq_len = int(pred_data.split("bp_")[0].split("_")[-1])
    for test_accession in test_accessions:
        read_preds_fgs_dict = remap_and_process_fgs_preds(test_accession, pred_data, seq_len)  # Test with the first accession

        with open(f'{project_root}/data/processed_data/predictions/processed_predictions/fgs_preds/{pred_data}/{test_accession}.pkl', "wb") as processed_preds_file:
            pickle.dump(read_preds_fgs_dict, processed_preds_file)