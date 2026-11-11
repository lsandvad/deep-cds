#!/bin/bash
# Step 0: copy the genome sequences needed for the CAMI/training overlap check to local disk.
#   - CAMI III source genomes: raw_data/CAMI_datasets/source_genomes.tar.gz (one 3 GB file, much
#     faster over sshfs than its 2,078 members), extracted locally
#   - train + val genomes: raw_data/genome_data/<acc>/<assembly>_genomic.fna, taken from the exact GCF_
#     folder when it exists, otherwise from the GCA_ folder with the same number and version
#     (partition lists use GCF_ accessions), plus their sequence_report.jsonl
#     (contig accessions, for the exact-assembly check), copied in parallel
#
# Usage: bash 00_stage_genomes.sh   (run inside screen; skips files already present)

set -euo pipefail
SRC=/tmp/nrt204/FragmentPredictor/data
WORK=${WORK:-/tmp/nrt204/cami_overlap_work}
NJOBS=${NJOBS:-32}
mkdir -p "$WORK/cami_genomes" "$WORK/train_val_genomes" "$WORK/partitions"

awk 1 "$SRC/processed_data/genome_partitions/train_partition_accessions.txt" | grep . > "$WORK/partitions/train.txt"  # no trailing newline in source
awk 1 "$SRC/processed_data/genome_partitions/val_partition_accessions.txt" | grep . > "$WORK/partitions/val.txt"
cat "$WORK/partitions/train.txt" "$WORK/partitions/val.txt" > "$WORK/partitions/train_val.txt"
echo "train/val genomes: $(wc -l < "$WORK/partitions/train_val.txt")"

# CAMI source genomes
if [ ! -f "$WORK/cami_genomes/.done" ]; then
    cp "$SRC/raw_data/CAMI_datasets/source_genomes.tar.gz" "$WORK/source_genomes.tar.gz"
    tar -xzf "$WORK/source_genomes.tar.gz" -C "$WORK/cami_genomes"
    rm "$WORK/source_genomes.tar.gz"
    touch "$WORK/cami_genomes/.done"
fi
echo "CAMI source genome files: $(find "$WORK/cami_genomes" -name '*.fna' | wc -l)"

# Train/val genomes (GCF_x -> folder GCA_x)
copy_genome() {
    gcf=$1; src=$2; dst=$3
    [ -s "$dst/$gcf.fna" ] && return 0
    for d in "$src/$gcf" "$src/GCA_${gcf#GCF_}"; do
        f=$(ls "$d"/*_genomic.fna 2>/dev/null | grep -v cds_from_genomic | head -1)
        if [ -n "$f" ]; then
            cp "$f" "$dst/$gcf.fna.part" && mv "$dst/$gcf.fna.part" "$dst/$gcf.fna"
            [ -f "$d/sequence_report.jsonl" ] && cp "$d/sequence_report.jsonl" "$dst/$gcf.sequence_report.jsonl"
            return 0
        fi
    done
    echo "MISSING genome FASTA: $gcf" >&2
    return 1
}
export -f copy_genome
xargs -P "$NJOBS" -I{} bash -c "copy_genome {} $SRC/raw_data/genome_data $WORK/train_val_genomes" < "$WORK/partitions/train_val.txt"
echo "train/val genome FASTAs: $(ls "$WORK/train_val_genomes" | grep -c '\.fna$') / $(wc -l < "$WORK/partitions/train_val.txt")"
echo "STAGING DONE"
