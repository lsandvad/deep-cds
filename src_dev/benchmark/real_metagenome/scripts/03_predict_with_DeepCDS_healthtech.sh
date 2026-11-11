#!/bin/bash
#SBATCH --job-name=poc_DeepCDS
#SBATCH --partition=gpu
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --gres=shard:1
#SBATCH --output=/home/projects2/lisani/DeepCDS/real_metagenome_poc/logs/slurm_DeepCDS_%x_%j.txt
#SBATCH --chdir=/home/projects2/lisani/DeepCDS/real_metagenome_poc/logs

# Real-metagenome POC: based on src_dev/benchmark/predict_metagenome/predict_with_DeepCDS_healthtech_S.sh,
# parametrized by DeepCDS variant; same conda env and batch size as the benchmark run.
# ERDA must already be mounted at MNT (and be visible on the compute node); the job does not mount it.
# Usage: sbatch 03_predict_with_DeepCDS_healthtech.sh <none|substitution|indel_substitution>
#        (another mount point: MNT=/path/to/FragmentPredictor sbatch --export=ALL 03_... <error_type>)
# Input: $MNT/real_metagenome_poc/fasta/deepcds/reads_all_cds_labels.fasta.gz (all reads, any length)
ERROR_TYPE=${1:?error type: none | substitution | indel_substitution}
MNT=${MNT:-/net/well/pool/projects2/lisani/DeepCDS/FragmentPredictor4}

source ~/.bashrc
conda activate pytorch-gpu-env

# Fail fast if the input is not visible from this node (mount missing, dead, or only on another node).
INPUT="$MNT/real_metagenome_poc/fasta/deepcds/reads_all_cds_labels.fasta.gz"
if [ ! -s "$INPUT" ]; then
    echo "ERROR: $INPUT not found on node $(hostname)." >&2
    echo "       ERDA is not mounted at $MNT here (sshfs mounts are node-local)." >&2
    ls -ld "$MNT" >&2 2>&1; ls "$MNT" 2>&1 | head -5 >&2
    exit 1
fi
echo "node $(hostname), input: $INPUT"

nvidia-smi
python3 /home/projects2/lisani/DeepCDS/src_dev/benchmark/real_metagenome/scripts/03_predict_with_DeepCDS_poc.py --project_path "$MNT" \
    --model "all_genomes" --error_type "$ERROR_TYPE" --sample reads_all --batch_size 256 --esm_model "8M" \
    --chunk_size 100000 --stride_aa 70 --amp off > "deepcds_${ERROR_TYPE}_reads_all.txt"
