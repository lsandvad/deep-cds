#!/bin/bash
# Download the UHGG v3.0 species-representative genomes (6,461) and concatenate them into one FASTA.
# Contig headers are prefixed with the species-rep accession: >MGYG000000001|<contig>
# Usage: bash 00b_download_uhgg.sh [WORK]
set -euo pipefail
WORK=${1:-/tmp/nrt204/real_metagenome_poc_work}
B=https://ftp.ebi.ac.uk/pub/databases/metagenomics/mgnify_genomes/human-gut/v3.0
mkdir -p "$WORK/uhgg/fna"
cd "$WORK/uhgg"
[ -f genomes-all_metadata.tsv ] || curl -sS -o genomes-all_metadata.tsv "$B/genomes-all_metadata.tsv"
awk -F'\t' 'NR>1 && $1==$14 {print $1}' genomes-all_metadata.tsv | sort -u > species_reps.txt
echo "species reps: $(wc -l < species_reps.txt)"

# species_catalogue/MGYG<first 10 chars>/MGYG.../genome/MGYG....fna.gz
# 8 parallel connections: at 32 the EBI FTP server starts refusing connections. Each file is written to a
# .part name and renamed only after a complete download, so an interrupted run leaves no truncated .fna.gz.
xargs -P 8 -I{} sh -c 'f=fna/{}.fna.gz; [ -s "$f" ] || { curl -sS --retry 10 --retry-connrefused --retry-delay 5 -o "$f.part" "'"$B"'/species_catalogue/$(echo {} | cut -c1-11)/{}/genome/{}.fna.gz" && mv "$f.part" "$f"; }' < species_reps.txt || true

n=$(ls fna/*.fna.gz | wc -l)
[ "$n" -eq "$(wc -l < species_reps.txt)" ] || { echo "only $n genomes downloaded" >&2; exit 1; }
ls fna/*.fna.gz | xargs -P 32 -n 50 gzip -t
echo "gzip integrity check passed for $n genomes"
while read -r g; do
    gzip -dc "fna/$g.fna.gz" | sed "s/^>/>$g|/"
done < species_reps.txt > uhgg_v3_species_reps.fna
grep -c '^>' uhgg_v3_species_reps.fna | sed 's/^/contigs: /'
awk '!/^>/ {n += length($0)} END {print "bp:", n}' uhgg_v3_species_reps.fna
