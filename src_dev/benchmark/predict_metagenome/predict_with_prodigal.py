import argparse
import glob
import os
import time

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Run Prodigal (v2.6.3) predictions on the CAMI metagenome test set")
parser.add_argument("--scarb_cluster", action="store_true",
                    help="Use SCARB cluster path (/tmp/nrt204/FragmentPredictor)")
args = parser.parse_args()

project_path = "/tmp/nrt204/FragmentPredictor" if args.scarb_cluster else "../../.."

data_dir = "CAMI_metagenome"
input_dir = f"{project_path}/data/processed_data/reads_processed/test/{data_dir}"
output_base = f"{project_path}/data/processed_data/predictions/raw_predictions/prodigal_preds/{data_dir}"

fasta_gz_paths = sorted(glob.glob(f"{input_dir}/*_cds_labels.fasta.gz"))
if not fasta_gz_paths:
    raise FileNotFoundError(f"No *_cds_labels.fasta.gz files found in: {input_dir}")

timing_log_path = f"{output_base}/timing_log.tsv"
os.makedirs(output_base, exist_ok=True)
with open(timing_log_path, "w") as timing_log:
    timing_log.write("sample\tn_reads\tgunzip_time_s\tprodigal_time_s\ttotal_time_s\n")

for gz_path in tqdm(fasta_gz_paths):
    sample = os.path.basename(gz_path).replace("_cds_labels.fasta.gz", "")
    print(sample, flush=True)

    path = f"{output_base}/{sample}"
    os.makedirs(path, exist_ok=True)

    sample_start = time.time()

    fasta_path = f"{input_dir}/{sample}_cds_labels.fasta"
    gunzip_start = time.time()
    os.system(f"gunzip -k {gz_path}")
    gunzip_time = time.time() - gunzip_start

    n_reads = sum(1 for _ in open(fasta_path)) // 2

    prodigal_start = time.time()
    os.system(f"prodigal -i {fasta_path} \
              -p meta -f gff -o {path}/{sample}.gff")
    prodigal_time = time.time() - prodigal_start

    #Clean up .gff output file
    os.system(f"grep '	+	' {path}/{sample}.gff > \
              {path}/{sample}+.gff")
    os.system(f"mv {path}/{sample}+.gff {path}/{sample}.gff")

    os.system(f"rm {fasta_path}")

    total_time = time.time() - sample_start
    print(f"{sample}: {n_reads} reads, gunzip {gunzip_time:.1f}s, prodigal {prodigal_time:.1f}s, "
          f"total {total_time:.1f}s ({n_reads / prodigal_time:.1f} reads/s)", flush=True)

    with open(timing_log_path, "a") as timing_log:
        timing_log.write(f"{sample}\t{n_reads}\t{gunzip_time:.2f}\t{prodigal_time:.2f}\t{total_time:.2f}\n")
