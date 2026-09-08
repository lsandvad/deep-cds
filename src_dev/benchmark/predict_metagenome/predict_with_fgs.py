import argparse
import glob
import os
import time

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Run FragGeneScanRs predictions on the CAMI metagenome test set")
parser.add_argument("--scarb_cluster", action="store_true",
                    help="Use SCARB cluster path (/tmp/nrt204/FragmentPredictor)")
args = parser.parse_args()

project_path = "/tmp/nrt204/FragmentPredictor" if args.scarb_cluster else "../../.."

data_dir = "CAMI_metagenome"
input_dir = f"{project_path}/data/processed_data/reads_processed/test/{data_dir}"
output_root = f"{project_path}/data/processed_data/predictions/raw_predictions/fgs_preds"

error_models = ["complete", "illumina_5", "illumina_10"]  #model for complete sequences (no errors), and models for 0.5% and 1.0% Illumina error rates

fasta_gz_paths = sorted(glob.glob(f"{input_dir}/*_cds_labels.fasta.gz"))
if not fasta_gz_paths:
    raise FileNotFoundError(f"No *_cds_labels.fasta.gz files found in: {input_dir}")

os.makedirs(output_root, exist_ok=True)
timing_log_path = f"{output_root}/{data_dir}_timing_log.tsv"
with open(timing_log_path, "w") as timing_log:
    timing_log.write("sample\terror_model\tn_reads\tgunzip_time_s\tfgs_time_s\ttotal_time_s\n")

for gz_path in tqdm(fasta_gz_paths):
    sample = os.path.basename(gz_path).replace("_cds_labels.fasta.gz", "")
    print(sample, flush=True)

    fasta_path = f"{input_dir}/{sample}_cds_labels.fasta"
    gunzip_start = time.time()
    os.system(f"gunzip -k {gz_path}")
    gunzip_time = time.time() - gunzip_start
    print(f"{sample}: gunzip took {gunzip_time:.1f}s", flush=True)

    n_reads = sum(1 for _ in open(fasta_path)) // 2

    #Run FragGeneScanRs once per error model, reusing the same decompressed fasta
    for error_model in error_models:
        print(f"Predicting with FGS error model: {error_model}", flush=True)

        path = f"{output_root}/{data_dir}_{error_model}/{sample}"
        os.makedirs(path, exist_ok=True)

        fgs_start = time.time()
        os.system(f"FragGeneScanRs -s {fasta_path} -t {error_model} \
                  -w 0 -o {path}/{sample}")
        fgs_time = time.time() - fgs_start
        total_time = gunzip_time + fgs_time

        print(f"{sample} ({error_model}): {n_reads} reads, fgs {fgs_time:.1f}s, "
              f"total {total_time:.1f}s ({n_reads / fgs_time:.1f} reads/s)", flush=True)

        with open(timing_log_path, "a") as timing_log:
            timing_log.write(f"{sample}\t{error_model}\t{n_reads}\t{gunzip_time:.2f}\t{fgs_time:.2f}\t{total_time:.2f}\n")

    os.system(f"rm {fasta_path}")
