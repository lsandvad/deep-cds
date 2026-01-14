import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import pickle

import torch


def mask_full_read(read, p=5e-4):
    """
    Mask nucleotides in the original read; ensures RF consistency.
    p = 0.05 % default
    """
    import random
    read_list = list(read)
    for i in range(len(read_list)):
        if random.random() < p:
            read_list[i] = "N"
    return "".join(read_list)


def mask_aa_sequence_from_nt(nt_seq, aa_seq):
    """Replace AA with <unk> if its codon contains N."""
    assert len(nt_seq) == 3 * len(aa_seq)
    aa_list = list(aa_seq)
    for i in range(len(aa_list)):
        codon = nt_seq[3*i:3*i+3]
        if "N" in codon:
            aa_list[i] = "<unk>"
    return "".join(aa_list)


def create_datasets_split(fold, 
                          archive, 
                          error_rates, 
                          seqs_len, 
                          model_type,
                          accession_filenames,
                          dataset_size,
                          data_subpath=data_subpath):
    
    # Load mapping depending on model_type (unchanged)
    if model_type == "model_with_errors":
        print("Creating dataset for model trained with sequencing errors...")
        with open(f'{data_subpath}/data/processed_data/model_data/shared_crf/model_with_errors/label_mappings/mapping_to_class.pkl', "rb") as mapping_file:
            mapping_dict_to_class = pickle.load(mapping_file)
    elif model_type == "model_with_substitution_errors":
        print("Creating dataset for model trained with substitution errors...")
        with open(f'{data_subpath}/data/processed_data/model_data/shared_crf/model_with_substitution_errors/label_mappings/mapping_to_class.pkl', "rb") as mapping_file:
            mapping_dict_to_class = pickle.load(mapping_file)
    elif model_type == "model_without_errors":
        print("Creating dataset for model trained without sequencing errors...")
        with open(f'{data_subpath}/data/processed_data/model_data/shared_crf/model_without_errors/label_mappings/mapping_to_class.pkl', "rb") as mapping_file:
            mapping_dict_to_class = pickle.load(mapping_file)
    else:
        raise ValueError("Invalid model type specified.")

    all_data_shared_rfs = []

    # Probability for inducing Ns (only used when fold == "train")
    p_unknown = 1e-4

    for accession_filename in tqdm(accession_filenames):
        accession = accession_filename.strip(".csv")
        processed_reads = pd.read_csv(
            f"{data_subpath}/data/processed_data/reads_processed/{archive}/{error_rates}/csv/{accession_filename}.csv.gz", 
            compression='gzip',
            index_col=0,
            low_memory=False
        )
        
        for _, row in processed_reads.iterrows():

            indel_detected = False
            indel_and_coding = False
            start_codon_detected = False
            stop_codon_detected = False

            seq_errors = str(row["indel_positions"])

            if "D" in seq_errors or "I" in seq_errors:
                indel_detected = True

            labels = {}

            # Build RF labels
            for frame in ["rf0", "rf1", "rf2"]:
                frame_labels = row[f"{frame}_labels"].replace(" ", "").replace("[", "").replace("]", "").split(",")
                labels[frame] = [int(label) for label in frame_labels]

                if indel_detected:
                    if 1 in labels[frame] and 0 in labels[frame]:

                        for i in range(1, len(labels[frame])):
                            if labels[frame][i] == 1 and labels[frame][i-1] == 0:
                                labels[frame][i] = 4

                        for i in range(len(labels[frame]) - 1):
                            if labels[frame][i] == 1 and labels[frame][i+1] == 0:
                                labels[frame][i] = 5
                        
                        indel_and_coding = True

                if 2 in labels[frame]:
                    start_codon_detected = True

                if 3 in labels[frame]:
                    stop_codon_detected = True

                # Extract nucleotide sequences for each RF
                if frame == "rf0":
                    nt_seq_rf0 = row["read"][:-3]
                elif frame == "rf1":
                    nt_seq_rf1 = row["read"][1:-2]
                elif frame == "rf2":
                    nt_seq_rf2 = row["read"][2:-1]

            # Determine sequence type (unchanged)
            if indel_and_coding:
                seq_type = "transition_indel"
            elif start_codon_detected and stop_codon_detected:
                seq_type = "transition_start_stop"
            elif start_codon_detected:
                seq_type = "transition_start"
            elif stop_codon_detected:
                seq_type = "transition_stop"
            elif row["cds_coords"] == "[]":
                seq_type = "non-coding"
            elif row["MD:Z"] != str(seqs_len) and error_rates != "without_errors":
                seq_type = "coding_with_substitutions"
            else:
                seq_type = "coding"

            # Encode labels
            label_encodings = []
            for i in range(len(labels["rf1"])):
                rfs_tuple = (labels["rf0"][i], labels["rf1"][i], labels["rf2"][i])
                label_encodings.append(mapping_dict_to_class[rfs_tuple])

            # Assertions unchanged
            assert len(label_encodings) == len(labels["rf0"][:-1]) == len(labels["rf1"]) == len(labels["rf2"])
            assert len(row['rf0_seq'][:-1]) == len(row['rf1_seq']) == len(row['rf2_seq'])
            assert len(nt_seq_rf0) == len(nt_seq_rf1) == len(nt_seq_rf2)

            # ------------------------------
            # *** AUGMENTATION STEP HERE ***
            # ------------------------------
            if fold == "train":
                # Mask the *original* read once
                masked_read = mask_full_read(row["read"], p=p_unknown)

                # Recompute the RF NT sequences *from the masked read*
                nt_seq_rf0 = masked_read[:-3]
                nt_seq_rf1 = masked_read[1:-2]
                nt_seq_rf2 = masked_read[2:-1]

                # Recompute AA sequences based on masked codons
                rf0_seq_aa = mask_aa_sequence_from_nt(nt_seq_rf0, row['rf0_seq'][:-1])
                rf1_seq_aa = mask_aa_sequence_from_nt(nt_seq_rf1, row['rf1_seq'])
                rf2_seq_aa = mask_aa_sequence_from_nt(nt_seq_rf2, row['rf2_seq'])
            else:
                nt_seq_rf0 = row["read"][:-3]
                nt_seq_rf1 = row["read"][1:-2]
                nt_seq_rf2 = row["read"][2:-1]

                rf0_seq_aa = row['rf0_seq'][:-1]
                rf1_seq_aa = row['rf1_seq']
                rf2_seq_aa = row['rf2_seq']
            # ------------------------------

            all_data_shared_rfs.append({
                'accession': accession,
                'rf0_seq_nt': nt_seq_rf0,
                'rf0_seq_aa': rf0_seq_aa,
                'rf0_labels': labels["rf0"][:-1],
                'rf1_seq_nt': nt_seq_rf1,
                'rf1_seq_aa': rf1_seq_aa,
                'rf1_labels': labels["rf1"],
                'rf2_seq_nt': nt_seq_rf2,
                'rf2_seq_aa': rf2_seq_aa,
                'rf2_labels': labels["rf2"],
                'label_encodings': label_encodings,
                'seq_desc': seq_type
            })


    os.makedirs(f"{data_subpath}/data/processed_data/model_data/shared_crf/{model_type}/datasets_model", exist_ok=True)

    processed_samples_shared_df = pd.DataFrame(all_data_shared_rfs)
    processed_samples_shared_df.to_csv(f"{data_subpath}/data/processed_data/model_data/shared_crf/{model_type}/datasets_model/{fold}{dataset_size}.csv.gz", index=False, compression="gzip")



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device_type = device.type  # "cuda", "mps", or "cpu"

    #For local runs
    if device_type != "cuda":
        data_subpath = "../.."

    #For runs on SCARB cluster
    elif device_type == "cuda":
        data_subpath = "/tmp/nrt204/FragmentPredictor" #Mount point on SCARB cluster


    train_accessions = open(f"{data_subpath}/data/processed_data/genome_partitions/train_partition_accessions.txt").read().splitlines()
    val_accessions = open(f"{data_subpath}/data/processed_data/genome_partitions/val_partition_accessions.txt").read().splitlines()

    train_accessions_400 = open(f"{data_subpath}/data/processed_data/genome_partitions/train_partition_accessions_400_genomes.txt").read().splitlines()
    train_accessions_200 = open(f"{data_subpath}/data/processed_data/genome_partitions/train_partition_accessions_200_genomes.txt").read().splitlines()
    train_accessions_100 = open(f"{data_subpath}/data/processed_data/genome_partitions/train_partition_accessions_100_genomes.txt").read().splitlines()
    seqs_len = 300

    ##### Generate datasets with sequencing errors ####
    for dataset in [train_accessions_400, train_accessions_200, train_accessions_100, train_accessions]:

        if dataset == train_accessions_400:
            dataset_size = "_400_genomes"
        elif dataset == train_accessions_200:
            dataset_size = "_200_genomes"
        elif dataset == train_accessions_100:
            dataset_size = "_100_genomes"
        else:
            dataset_size = "_all_genomes"

        create_datasets_split("train", "train_val", "with_errors", seqs_len, "model_with_errors", dataset, dataset_size)

    #Generate validation dataset
    create_datasets_split("val", "train_val", "with_errors", seqs_len, "model_with_errors", val_accessions, "")



    #### Generate datasets with substitution errors ####
    for dataset in [train_accessions_400, train_accessions_200, train_accessions_100, train_accessions]:

        if dataset == train_accessions_400:
            dataset_size = "_400_genomes"
        elif dataset == train_accessions_200:
            dataset_size = "_200_genomes"
        elif dataset == train_accessions_100:
            dataset_size = "_100_genomes"
        else:
            dataset_size = "_all_genomes"

        create_datasets_split("train", "train_val", "with_substitution_errors", seqs_len, "model_with_substitution_errors", dataset, dataset_size)

    #Generate validation dataset
    create_datasets_split("val", "train_val", "with_substitution_errors", seqs_len, "model_with_substitution_errors", val_accessions, "")



    ##### Generate datasets without sequencing errors ####
    for dataset in [train_accessions_100, train_accessions_200, train_accessions_400, train_accessions]:
        if dataset == train_accessions_400:
            dataset_size = "_400_genomes"
        elif dataset == train_accessions_200:
            dataset_size = "_200_genomes"
        elif dataset == train_accessions_100:
            dataset_size = "_100_genomes"
        else:
            dataset_size = "_all_genomes"

        create_datasets_split("train", "train_val", "without_errors", seqs_len, "model_without_errors", dataset, dataset_size)

    #Generate validation dataset
    create_datasets_split("val", "train_val", "without_errors", seqs_len, "model_without_errors", val_accessions, "")