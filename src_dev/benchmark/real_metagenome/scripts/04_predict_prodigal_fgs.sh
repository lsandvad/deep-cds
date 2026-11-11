#!/bin/bash
# Step 4 (SCARB): MetaProdigal and FragGeneScanRs on the reads >= 61 bp, with the benchmark commands
# (src_dev/benchmark/predict_metagenome/predict_with_prodigal.py / predict_with_fgs.py):
#   prodigal -i <fa> -p meta -f gff -o <gff>
#   FragGeneScanRs -s <fa> -t <complete|illumina_5|illumina_10> -w 0 -o <prefix>
# The input is split into NCHUNK files run in parallel; every read is predicted independently, so the
# chunking does not change any prediction.
# -q (Prodigal) only silences its progress messages on stderr. Outputs: preds/<tool>/chunk_NNN.gff.gz
# Usage: bash 04_predict_prodigal_fgs.sh [WORK] [NCHUNK]
set -euo pipefail
WORK=${1:-/tmp/nrt204/real_metagenome_poc_work}
NCHUNK=${2:-128}
BIN=~/.conda/envs/gene_prediction_env/bin
cd "$WORK"
mkdir -p chunks preds/prodigal preds/fgs_complete preds/fgs_illumina_5 preds/fgs_illumina_10
$BIN/prodigal -v 2>&1 | grep -i prodigal
$BIN/FragGeneScanRs --version

if [ ! -f chunks/done ]; then
    pigz -dc fasta/reads_ge61.fa.gz | awk -v n="$NCHUNK" '/^>/ {f = sprintf("chunks/chunk_%03d.fa", (i++) % n)} {print > f}'
    touch chunks/done
fi

export BIN
run_chunk() {
    c=$(basename "$1" .fa)
    $BIN/prodigal -i "$1" -p meta -f gff -o preds/prodigal/$c.gff -q
    gzip -f preds/prodigal/$c.gff   # GFFs are gzipped right away (disk)
    for m in complete illumina_5 illumina_10; do
        $BIN/FragGeneScanRs -s "$1" -t $m -w 0 -o preds/fgs_$m/$c
        rm -f preds/fgs_$m/$c.out preds/fgs_$m/$c.faa preds/fgs_$m/$c.ffn   # only the .gff is used (disk)
        gzip -f preds/fgs_$m/$c.gff
    done
}
export -f run_chunk
SECONDS=0
ls chunks/chunk_*.fa | xargs -P "$NCHUNK" -I{} bash -c 'run_chunk {}'
echo "prodigal + fgs wall time: ${SECONDS}s"
for d in prodigal fgs_complete fgs_illumina_5 fgs_illumina_10; do
    echo "$d: $(zcat preds/$d/*.gff.gz | grep -vc '^#') CDS lines"
done
