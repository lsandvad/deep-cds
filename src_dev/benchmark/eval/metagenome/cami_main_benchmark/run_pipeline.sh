#!/bin/bash
# Run the CAMI benchmark (steps 0-5). Results are then analysed in cami_benchmark_results.ipynb.
# Step 4 needs the train/val fragment DB from ../../without_errors/homology_audit (its steps 1-2).
# Usage: screen -dmS cami_benchmark bash run_pipeline.sh     (logs in $WORK/logs/)
set -euo pipefail
cd "$(dirname "$0")"
export WORK=${WORK:-/tmp/nrt204/cami_overlap_work}
PY=/home/nrt204/.conda/envs/gene_prediction_env/bin/python
LOGS=$WORK/logs
mkdir -p "$LOGS"

bash 00_stage_genomes.sh                          > "$LOGS/00_stage.log" 2>&1
$PY 01_per_bin_counts.py                          > "$LOGS/01_counts.log" 2>&1
for v in CAMI_metagenome CAMI_metagenome_no_ab_initio CAMI_metagenome_no_cds_contigs; do
    $PY 01_per_bin_counts.py --variant "$v"       > "$LOGS/01_counts_$v.log" 2>&1
done
$PY 02_genome_overlap.py --work "$WORK"           > "$LOGS/02_ani.log" 2>&1
$PY 03_extract_cami_fragments.py --work "$WORK"   > "$LOGS/03_extract.log" 2>&1
bash 04_search_cami_fragments.sh                  > "$LOGS/04_search.log" 2>&1
$PY 05_assign_bins.py --work "$WORK"              > "$LOGS/05_bins.log" 2>&1
echo "PIPELINE DONE - open cami_benchmark_results.ipynb"
