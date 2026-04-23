import os
from collections import defaultdict

import numpy as np
import pandas as pd

from tqdm import tqdm

"""
Count reads in each dataset; both train and test for each error type.
Process for supplementary information for the paper (store as .csv files), and also to have an idea of how many reads are in each dataset for training and evaluation purposes.
"""

mount_point = "/tmp/nrt204/FragmentPredictor" #local "../.."
os.makedirs(os.path.join(mount_point, "data/processed_data/dataset_information/read_counts_datasets"), exist_ok = True)

def load_tax_info():
    """Load taxonomic info and partition info for all genomes in the dataset."""

    tax_info_df = pd.read_csv(os.path.join(mount_point, "data/processed_data/dataset_information/genomes_info_with_partitions.csv"))
    return tax_info_df

def count_reads(partition_type, data_type, tax_info_df):
    """Count number of reads in each dataset, and store in a .csv file with taxonomic and partition info for each genome."""

    dict_info = dict()

    # Get list of accession numbers for the given partition type and data type
    reads_info_path = os.path.join(mount_point, f"data/processed_data/reads_processed/{partition_type}/{data_type}/csv")
    acc_numbers = os.listdir(reads_info_path)
    acc_numbers = [acc for acc in acc_numbers if acc.endswith(".csv.gz")]

    # Process each dataset, count number of reads, and get taxonomic and partition info for the corresponding genome
    for acc_number in tqdm(acc_numbers):
        reads_df = pd.read_csv(reads_info_path + "/" + acc_number, compression = "gzip")
        n_reads = len(reads_df)
        acc_number_clean = acc_number.strip(".csv.gz")

        tax_row = tax_info_df[tax_info_df["accession"] == acc_number_clean]
        organism_name = tax_row["species"].values[0]
        partition = tax_row["partition"].values[0]

        if acc_number not in dict_info.keys():
            dict_info[acc_number_clean] = dict()
            dict_info[acc_number_clean]["species"] = organism_name
            dict_info[acc_number_clean]["n_reads"] = n_reads
            dict_info[acc_number_clean]["partition"] = partition
    
    # Store info in a .csv file
    dict_info_df = pd.DataFrame.from_dict(dict_info, orient = "index", columns = ["species", "partition", "n_reads"])
    dict_info_df.index.name = "accession_number"
    dict_info_df.to_csv(os.path.join(mount_point, f"data/processed_data/dataset_information/read_counts_datasets/count_reads_{partition_type}_{data_type}.csv"))

#Load taxonomic and parittion info
tax_info_df = load_tax_info()

#Generate read count statistics for train and val datasets
count_reads("train_val", "with_errors", tax_info_df)
count_reads("train_val", "without_errors", tax_info_df)
count_reads("train_val", "with_substitution_errors", tax_info_df)

#Generate read count statistics for test datasets
test_dirs = os.listdir(os.path.join(mount_point, "data/processed_data/reads_processed/test"))
test_dirs = [test_dir for test_dir in test_dirs if test_dir != ".DS_Store"]

for test_dir in test_dirs:
    print(test_dir)
    count_reads("test", test_dir, tax_info_df)
