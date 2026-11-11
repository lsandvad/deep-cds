#!/bin/bash
# Step 1: fastp QC (defaults, paired mode) -> Bowtie2 host filter against GRCh38 (no-alt analysis set).
# Pairs are kept together through QC; host filtering keeps a pair only if NEITHER mate aligns to GRCh38
# (samtools fastq -f 12: read unmapped AND mate unmapped), so R1 and R2 stay in sync before they are split into separate reads in 02_make_fasta.py.
# Usage: bash 01_qc_host_filter.sh [WORK] [THREADS]
set -euo pipefail
WORK=${1:-/tmp/nrt204/real_metagenome_poc_work}
T=${2:-64}
RUN=ERR321493
ENV=~/.conda/envs/poc_env/bin
cd "$WORK"
mkdir -p qc host

echo "== md5 (ENA: 0b678230496c602100c9ef4d4e6a3868 / 47d98501d7cde3d05f819fb96f077edd)"
md5sum raw/${RUN}_1.fastq.gz raw/${RUN}_2.fastq.gz

echo "== fastp"
$ENV/fastp --version 2>&1
$ENV/fastp -i raw/${RUN}_1.fastq.gz -I raw/${RUN}_2.fastq.gz \
    -o qc/${RUN}_1.qc.fastq.gz -O qc/${RUN}_2.qc.fastq.gz \
    -w 16 -j qc/fastp.json -h qc/fastp.html

echo "== bowtie2 host filter"
$ENV/bowtie2 --version | head -1
$ENV/samtools --version | head -1
$ENV/bowtie2 -p "$T" -x grch38/GRCh38_noalt_as \
    -1 qc/${RUN}_1.qc.fastq.gz -2 qc/${RUN}_2.qc.fastq.gz \
    2> host/bowtie2_grch38.log \
  | $ENV/samtools fastq -@ 8 -f 12 -F 256 -n -1 host/${RUN}_1.nohost.fastq.gz -2 host/${RUN}_2.nohost.fastq.gz \
      -0 /dev/null -s /dev/null -
cat host/bowtie2_grch38.log
for i in 1 2; do
    echo "host-filtered R$i reads: $(( $(gzip -dc host/${RUN}_$i.nohost.fastq.gz | wc -l) / 4 ))"
done
