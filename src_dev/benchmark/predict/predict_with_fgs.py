import pandas as pd
from tqdm import tqdm
import os 

#project_path = f"/tmp/nrt204/FragmentPredictor" #On SCARB cluster; mount ERDA path
project_path = "../../.." #Local pc
test_accessions = open(f"{project_path}/data/processed_data/genome_partitions/test_partition_accessions.txt").read().splitlines()


########Predict on test sets without sequencing errors########
error_models = ["complete"] #model developed for complete sequences (no errors)

base_path = f"{project_path}/data/processed_data/reads_processed/test/"
data_dirs = os.listdir(base_path)
data_dirs = [d for d in data_dirs
                  if d.startswith("without_errors")]

for error_model in error_models:
    for data_dir in data_dirs:

        print(f"Predicting with error model: {error_model} and on test set: {data_dir}]")

        for test_accession in tqdm(test_accessions):
            #Run FragGeneScanRs
            path = f"{project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}"
            os.makedirs(path, exist_ok=True)    

            gz_path = f"{project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta.gz"
            if not os.path.isfile(gz_path):
                raise FileNotFoundError(f"GZIP file not found: {gz_path}")
            os.system(f"gunzip -k {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta.gz")
            os.system(f"FragGeneScanRs -s {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta -t {error_model} -w 0 -o {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}")

            #Clean up .gff output file 
            os.system(f"grep '	+	' {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}.gff > {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}+.gff")
            os.system(f"mv {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}+.gff {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}.gff")

            #Clean up .out output file
            os.system(f"grep -B 1 '	+	' {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}.out | grep -v '^--$' > {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}+.out")
            os.system(f"mv {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}+.out {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}.out")

            #Clean up .faa output file
            os.system(f"grep -A 1 '_+' {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}.faa | grep -v '^--$' > {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}+.faa")
            os.system(f"mv {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}+.faa {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}.faa")


            #Clean out .ffn output file
            os.system(f"grep -A 1 '_+' {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}.ffn | grep -v '^--$' > {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}+.ffn")
            os.system(f"mv {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}+.ffn {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}/{test_accession}/{test_accession}.ffn")

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
                raise FileNotFoundError(f"GZIP file not found: {gz_path}")
            
            os.system(f"gunzip -k {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta.gz")
            os.system(f"FragGeneScanRs -s {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta -t {error_model} -w 0 -o {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}")

            #Clean up .gff output file
            os.system(f"grep '	+	' {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}.gff > {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}+.gff")
            os.system(f"mv {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}+.gff {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}.gff")

            #Clean up .out output file
            os.system(f"grep -B 1 '	+	' {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}.out | grep -v '^--$' > {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}+.out")
            os.system(f"mv {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}+.out {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}.out")

            #Clean up .faa output file
            os.system(f"grep -A 1 '_+' {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}.faa | grep -v '^--$' > {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}+.faa")
            os.system(f"mv {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}+.faa {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}.faa")

            #Clean out .ffn output file
            os.system(f"grep -A 1 '_+' {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}.ffn | grep -v '^--$' > {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}+.ffn")
            os.system(f"mv {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}+.ffn {project_path}/data/processed_data/predictions/raw_predictions/fgs_preds/{data_dir}_{error_model}/{test_accession}/{test_accession}.ffn")

            os.system(f"rm {project_path}/data/processed_data/reads_processed/test/{data_dir}/fasta/{test_accession}.fasta")