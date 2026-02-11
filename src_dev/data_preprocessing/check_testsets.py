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

testset_type = "without_errors_300bp"
poor_accessions = []

for test_accession in test_accessions:
    coding_reads = 0
    non_coding_reads = 0

    # Load test dataset
    test_data_df = pd.read_csv(f"{project_root}/data/processed_data/reads_processed/test/{testset_type}/csv/{test_accession}.csv.gz", index_col=0, compression="gzip")
    
    # Get all read names in test set 
    all_test_read_names = list(set(list(test_data_df["read_name"])))

    # Process each read
    for _, row in test_data_df.iterrows():
        cds_coords = row["cds_coords"]

        if cds_coords == "[]":
            non_coding_reads += 1
        else:
            coding_reads += 1

    if coding_reads / (coding_reads + non_coding_reads) * 100 < 10:
        poor_accessions.append(test_accession)
        print(f"Test accession: {test_accession}, coding percentage reads: {coding_reads / (coding_reads + non_coding_reads) * 100:.2f}%")
        

print("Poor accessions:", poor_accessions)