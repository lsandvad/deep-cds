import argparse
import os

import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Run Prodigal (v2.6.3) predictions on test sets")
parser.add_argument("--scarb_cluster", action="store_true",
                    help="Use SCARB cluster path (/tmp/nrt204/FragmentPredictor)")
parser.add_argument("--predict_without_errors", action="store_true",
                    help = "Predict on datasets without sequencing errors")
parser.add_argument("--predict_with_errors", action="store_true",
                    help = "Predict on datasets with sequencing errors")
args = parser.parse_args()

project_path = "/tmp/nrt204/FragmentPredictor" if args.scarb_cluster else "../../.."
with open(f"{project_path}/data/processed_data/genome_partitions/test_partition_accessions.txt") as f:
    test_accessions = f.read().splitlines()

#Get test accession files
test_accessions = open(f"{project_path}/data/processed_data/genome_partitions/test_partition_accessions.txt").read().splitlines()

#Initialize
data_dirs_no_errors = []
data_dirs_errors = []

# Get testset directories for testsets without sequencing errors
if args.predict_without_errors:

    base_path = f"{project_path}/data/processed_data/reads_processed/test/"
    data_dirs_no_errors = os.listdir(base_path)
    data_dirs_no_errors = [d for d in data_dirs_no_errors
                    if d.startswith("without_errors")]
    data_dirs_no_errors = [d for d in data_dirs_no_errors
                    if not d.endswith("30bp")]

# Get testset directories for testsets with sequencing errors
if args.predict_with_errors:
    base_path = f"{project_path}/data/processed_data/reads_processed/test/"
    data_dirs_errors = os.listdir(base_path)
    data_dirs_errors = [d for d in data_dirs_errors
                    if d.startswith("with_errors")]

#Merge dirs to predict on
data_dirs = data_dirs_no_errors + data_dirs_errors


#Run MetaProdigal on all specificed test sets 
for data_dir in data_dirs:
    for test_accession in tqdm(test_accessions):
        path = f"{project_path}/data/processed_data/predictions/raw_predictions/prodigal_preds/{data_dir}/{test_accession}"
        os.makedirs(path, exist_ok=True)    

        gz_path = f"{project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta.gz"
        if not os.path.isfile(gz_path):
            raise FileNotFoundError(f"gzip file not found: {gz_path}")
            
        os.system(f"gunzip -k {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta.gz")
        os.system(f"prodigal -i {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta \
                  -p meta -f gff -o {project_path}/data/processed_data/predictions/raw_predictions/prodigal_preds/{data_dir}/{test_accession}/{test_accession}.gff")

        #Clean up .gff output file 
        os.system(f"grep '	+	' {project_path}/data/processed_data/predictions/raw_predictions/prodigal_preds/{data_dir}/{test_accession}/{test_accession}.gff > \
                  {project_path}/data/processed_data/predictions/raw_predictions/prodigal_preds/{data_dir}/{test_accession}/{test_accession}+.gff")
        os.system(f"mv {project_path}/data/processed_data/predictions/raw_predictions/prodigal_preds/{data_dir}/{test_accession}/{test_accession}+.gff \
                  {project_path}/data/processed_data/predictions/raw_predictions/prodigal_preds/{data_dir}/{test_accession}/{test_accession}.gff")

        os.system(f"rm {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta")