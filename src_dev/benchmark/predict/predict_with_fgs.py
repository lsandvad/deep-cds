import argparse
import os

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Run FragGeneScanRs predictions on test sets")
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

########Predict on test sets without sequencing errors########
if args.predict_without_errors:
    error_models = ["complete"] #model developed for complete sequences (no errors)

    base_path = f"{project_path}/data/processed_data/reads_processed/test/"
    data_dirs = os.listdir(base_path)
    data_dirs = [d for d in data_dirs
                    if d.startswith("without_errors")]
    data_dirs = [d for d in data_dirs
                    if not d.endswith("30bp")]

    for error_model in error_models:
        for data_dir in data_dirs:

            print(f"Predicting with error model {error_model}, and on test set: {data_dir}")

            for test_accession in tqdm(test_accessions):
                #Run FragGeneScanRs
                path = f"{project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}"
                os.makedirs(path, exist_ok=True)

                gz_path = f"{project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta.gz"
                if not os.path.isfile(gz_path):
                    raise FileNotFoundError(f"gzip file not found: {gz_path}")
                os.system(f"gunzip -k {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta.gz")
                os.system(f"FragGeneScanRs -s {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta -t {error_model} \
                          -w 0 -o {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}")

                #Remove decompressed fasta file
                os.system(f"rm {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta")




########Predict on test sets with sequencing errors########
if args.predict_with_errors:
    error_models = ["complete", "illumina_5", "illumina_10"] #try model developed for no sequencing errors and models developed for Illumina errors

    base_path = f"{project_path}/data/processed_data/reads_processed/test/"
    data_dirs = os.listdir(base_path)
    data_dirs = [d for d in data_dirs
                    if d.startswith("with_errors")]
    data_dirs += ["MiSeq_v3_300bp", "HiSeq2500_150bp", "NextSeq500_150bp"] #add test sets with simulated modern sequencing errors based on art_modern instead of Mason

    for error_model in error_models:
        for data_dir in data_dirs:

            print(f"Predicting with FGS error model: {error_model} and on test set: {data_dir}", flush = True)

            for test_accession in tqdm(test_accessions):
                #Run FragGeneScanRs
                path = f"{project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}"
                os.makedirs(path, exist_ok = True)

                gz_path = f"{project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta.gz"
                if not os.path.isfile(gz_path):
                    raise FileNotFoundError(f"gzip file not found: {gz_path}")

                os.system(f"gunzip -k {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta.gz")
                os.system(f"FragGeneScanRs -s {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta -t {error_model} \
                          -w 0 -o {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}")


                #Remove decompressed fasta file
                os.system(f"rm {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta")
