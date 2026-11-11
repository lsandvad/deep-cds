#!/bin/bash
#
# Run MetaGeneMark-2 on all read-based test sets (all 215 test genomes each) and postprocess
# the predictions into the benchmark format.
#
# MetaGeneMark-2: https://github.com/gatech-genemark/MetaGeneMark-2, commit 4bc3434
# (GeneMark.hmm-2 v1.23, run_mgm2.pl v1.01; CC BY-NC-SA 4.0), compiled in
# /home/nrt204/tools/MetaGeneMark-2/src against Boost from conda env mgm2_env.
# Default settings via run_mgm2.pl (both strands; incomplete genes allowed at both sequence ends).
# run_mgm2.pl assigns genetic code 11 to every sequence < 5,000 bp, i.e. to all reads.
#
# Input:  reads_processed/test/{SET}/fasta/{acc}.fasta.gz   (the files Prodigal/FGS were run on)
# Output: predictions/raw_predictions/metagenemark2_preds/{SET}/{acc}/{acc}.gff3.gz
#         predictions/processed_predictions/metagenemark2_preds/{SET}/{acc}/model_preds_dict.pkl
#
# Each set is staged to local disk, predicted, postprocessed, uploaded to ERDA in parallel and
# then removed locally. Sets already marked done are skipped, so the script can be re-run.
#
# Usage: screen -dmS mgm2_all bash predict_with_metagenemark2.sh [SET ...]   (default: all sets)

set -euo pipefail

PROJECT_ROOT=/tmp/nrt204/FragmentPredictor
DATA=$PROJECT_ROOT/data/processed_data
MGM2=/home/nrt204/tools/MetaGeneMark-2/src
PY=/home/nrt204/.conda/envs/gene_prediction_env/bin/python
WORK=${WORK:-/tmp/nrt204/mgm2_all_work}
NJOBS=${NJOBS:-96}
NCOPY=${NCOPY:-32}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTPROCESS=$SCRIPT_DIR/../postprocess/postprocess_metagenemark2_preds.py

SETS=("$@")
if [ ${#SETS[@]} -eq 0 ]; then
    SETS=(without_errors_60bp without_errors_75bp without_errors_100bp without_errors_150bp
          without_errors_300bp without_errors_700bp without_errors_1000bp
          with_errors_5e-06i_0.004s_60bp with_errors_5e-06i_0.004s_75bp with_errors_5e-06i_0.004s_100bp
          with_errors_5e-06i_0.004s_150bp with_errors_5e-06i_0.004s_300bp
          with_errors_1.25e-05i_0.01s_60bp with_errors_1.25e-05i_0.01s_75bp with_errors_1.25e-05i_0.01s_100bp
          with_errors_1.25e-05i_0.01s_150bp with_errors_1.25e-05i_0.01s_300bp
          with_errors_3.75e-05i_0.03s_60bp with_errors_3.75e-05i_0.03s_75bp with_errors_3.75e-05i_0.03s_100bp
          with_errors_3.75e-05i_0.03s_150bp with_errors_3.75e-05i_0.03s_300bp
          HiSeq2500_150bp MiSeq_v3_300bp NextSeq500_150bp)
fi

mkdir -p "$WORK"
ACC_FILE=$WORK/test_accessions.txt
awk 1 "$DATA/genome_partitions/test_partition_accessions.txt" | grep . > "$ACC_FILE"  # source file lacks a trailing newline
N_ACC=$(wc -l < "$ACC_FILE")

copy() {  # copy src dst, atomically
    mkdir -p "$(dirname "$2")"
    [ -s "$2" ] && return 0
    cp "$1" "$2.part" && mv "$2.part" "$2"
}
run_one() {  # acc set_dir
    acc=$1; d=$2
    [ -s "$d/raw/$acc.gff3.gz" ] && return 0
    zcat "$d/fasta/$acc.fasta.gz" > "$d/fasta/$acc.fasta"
    perl "$MGM2/run_mgm2.pl" --seq "$d/fasta/$acc.fasta" --out "$d/raw/$acc.gff3" \
        --format gff3 --clean --tmpf "$d/tmp/$acc" > /dev/null
    rm -f "$d/fasta/$acc.fasta"
}
export -f copy run_one
export MGM2

echo "MetaGeneMark-2 on ${#SETS[@]} test sets x $N_ACC genomes; local work dir $WORK"
for SET in "${SETS[@]}"; do
    D=$WORK/$SET
    if [ -f "$D/DONE" ]; then echo "[skip] $SET (done)"; continue; fi
    echo "=== $SET  $(date)"
    mkdir -p "$D/fasta" "$D/raw" "$D/tmp" "$D/processed"

    # 1. stage inputs
    xargs -P "$NCOPY" -I{} bash -c "copy $DATA/reads_processed/test/$SET/fasta/{}.fasta.gz $D/fasta/{}.fasta.gz" < "$ACC_FILE"

    # 2. predict (gff3 written uncompressed by run_mgm2.pl, gzipped right after postprocessing)
    xargs -P "$NJOBS" -I{} bash -c "run_one {} $D" < "$ACC_FILE"
    n_raw=$(ls "$D/raw" | grep -c '\.gff3$' || true)
    [ "$n_raw" -eq "$N_ACC" ] || { echo "ERROR: $SET has $n_raw / $N_ACC GFF3 files"; exit 1; }

    # 3. postprocess
    $PY "$POSTPROCESS" --raw_dir "$D/raw" --out_dir "$D/processed" --accessions "$ACC_FILE"
    ls "$D/raw"/*.gff3 | xargs -P "$NJOBS" -n 1 gzip -f

    # 4. upload raw + processed in parallel
    while read -r acc; do
        echo "$D/raw/$acc.gff3.gz $DATA/predictions/raw_predictions/metagenemark2_preds/$SET/$acc/$acc.gff3.gz"
        echo "$D/processed/$acc/model_preds_dict.pkl $DATA/predictions/processed_predictions/metagenemark2_preds/$SET/$acc/model_preds_dict.pkl"
    done < "$ACC_FILE" | xargs -P "$NCOPY" -L 1 bash -c 'copy "$0" "$1"'

    # 5. verify upload, then clean up local copies
    n_up=$(find "$DATA/predictions/processed_predictions/metagenemark2_preds/$SET" -name model_preds_dict.pkl | wc -l)
    n_raw_up=$(find "$DATA/predictions/raw_predictions/metagenemark2_preds/$SET" -name '*.gff3.gz' | wc -l)
    [ "$n_up" -eq "$N_ACC" ] && [ "$n_raw_up" -eq "$N_ACC" ] || { echo "ERROR: $SET uploaded $n_up / $n_raw_up of $N_ACC"; exit 1; }
    rm -rf "$D/fasta" "$D/raw" "$D/tmp"
    touch "$D/DONE"
    echo "    done: $SET  $(date)"
done
echo "ALL READ SETS DONE"
