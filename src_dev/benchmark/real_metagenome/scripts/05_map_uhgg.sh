#!/bin/bash
# Step 3: map every read >= 61 bp (R1 and R2 as separate, unpaired reads) to the UHGG v3.0 species
# representatives with Bowtie2 --very-sensitive-local. One (best) alignment per read is reported; --reorder keeps the BAM in FASTA order.
# Output: WORK/mapping/uhgg.bam (primary alignments + unmapped reads; all reads are kept)
# Usage: bash 05_map_uhgg.sh [WORK] [THREADS]
set -euo pipefail
WORK=${1:-/tmp/nrt204/real_metagenome_poc_work}
T=${2:-96}
ENV=~/.conda/envs/poc_env/bin
cd "$WORK"
mkdir -p mapping uhgg/index
$ENV/bowtie2 --version | head -1
$ENV/samtools --version | head -1

# The multithreaded build writes its suffix-array buckets to disk (> 100 GB for 15 Gbp), which does not fit
# on /tmp; it is built in RAM (/dev/shm) and only the final index (~20 GB) is moved to WORK.
if [ ! -f uhgg/index/done ]; then
    SECONDS=0
    BUILD=/dev/shm/nrt204_uhgg_index
    rm -rf "$BUILD"; mkdir -p "$BUILD"
    $ENV/bowtie2-build --large-index --threads "$T" uhgg/uhgg_v3_species_reps.fna "$BUILD/uhgg" > uhgg/index/build.log 2>&1
    mv "$BUILD"/uhgg.*.bt2l uhgg/index/
    rm -rf "$BUILD"
    echo "bowtie2-build wall time: ${SECONDS}s"
    touch uhgg/index/done
fi

SECONDS=0
$ENV/bowtie2 --very-sensitive-local --reorder -p "$T" -x uhgg/index/uhgg -f -U fasta/reads_ge61.fa.gz \
    2> mapping/bowtie2_uhgg.log \
  | $ENV/samtools view -@ 8 -b -F 0x900 -o mapping/uhgg.bam -
echo "bowtie2 wall time: ${SECONDS}s"
cat mapping/bowtie2_uhgg.log
