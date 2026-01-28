import argparse
import os

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Run FragGeneScanRs predictions on test sets")
parser.add_argument("--scarb_cluster", action="store_true",
                    help="Use SCARB cluster path (/tmp/nrt204/FragmentPredictor)")
args = parser.parse_args()

project_path = "/tmp/nrt204/FragmentPredictor" if args.scarb_cluster else "../../.."
with open(f"{project_path}/data/processed_data/genome_partitions/test_partition_accessions.txt") as f:
    test_accessions = f.read().splitlines()


########Predict on test sets without sequencing errors########
error_models = ["complete"] #model developed for complete sequences (no errors)

base_path = f"{project_path}/data/processed_data/reads_processed/test/"
data_dirs = os.listdir(base_path)
data_dirs = [d for d in data_dirs
                  if d.startswith("without_errors")]

for error_model in error_models:
    for data_dir in data_dirs:

        print(f"Predicting with error model: {error_model} and on test set: {data_dir}")

        for test_accession in tqdm(test_accessions):
            #Run FragGeneScanRs
            path = f"{project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}"
            os.makedirs(path, exist_ok=True)    

            gz_path = f"{project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta.gz"
            if not os.path.isfile(gz_path):
                raise FileNotFoundError(f"gzip file not found: {gz_path}")
            os.system(f"gunzip -k {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta.gz")
            os.system(f"FragGeneScanRs -s {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta -t {error_model} -w 0 -o {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}")

            #Filter .gff to keep only forward strand (+) predictions
            os.system(f"grep '	+	' {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}.gff > {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}+.gff")
            os.system(f"mv {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}+.gff {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}.gff")

            #Filter .out to keep only forward strand (+) predictions
            os.system(f"grep -B 1 '	+	' {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}.out | grep -v '^--$' > {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}+.out")
            os.system(f"mv {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}+.out {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}.out")

            #Filter .faa to keep only forward strand (+) predictions
            os.system(f"grep -A 1 '_+' {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}.faa | grep -v '^--$' > {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}+.faa")
            os.system(f"mv {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}+.faa {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}.faa")


            #Filter .ffn to keep only forward strand (+) predictions
            os.system(f"grep -A 1 '_+' {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}.ffn | grep -v '^--$' > {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}+.ffn")
            os.system(f"mv {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}+.ffn {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}.ffn")

            #Remove decompressed fasta file
            os.system(f"rm {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta")




########Predict on test sets with sequencing errors########
error_models = ["complete", "illumina_5", "illumina_10"] #try model developed for no sequencing errors and models developed for Illumina errors 

base_path = f"{project_path}/data/processed_data/reads_processed/test/"
data_dirs = os.listdir(base_path)
data_dirs = [d for d in data_dirs
                  if d.startswith("with_errors")]

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
            os.system(f"FragGeneScanRs -s {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta -t {error_model} -w 0 -o {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}")

            #Filter .gff to keep only forward strand (+) predictions
            os.system(f"grep '	+	' {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}.gff > {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}+.gff")
            os.system(f"mv {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}+.gff {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}.gff")

            #Filter .out to keep only forward strand (+) predictions
            os.system(f"grep -B 1 '	+	' {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}.out | grep -v '^--$' > {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}+.out")
            os.system(f"mv {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}+.out {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}.out")

            #Filter .faa to keep only forward strand (+) predictions
            os.system(f"grep -A 1 '_+' {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}.faa | grep -v '^--$' > {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}+.faa")
            os.system(f"mv {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}+.faa {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}.faa")

            #Filter .ffn to keep only forward strand (+) predictions
            os.system(f"grep -A 1 '_+' {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}.ffn | grep -v '^--$' > {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}+.ffn")
            os.system(f"mv {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}+.ffn {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}.ffn")

            #Remove decompressed fasta file
            os.system(f"rm {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta")