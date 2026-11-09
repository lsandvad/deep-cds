#!/bin/bash
# Runs the homology audit (steps 0-10); the results are then analysed in homology_audit_results.ipynb.
# Intermediate files go to $WORK (local disk, ~30 GB), result tables to $RESULTS (ERDA).
# Usage: setsid nohup bash run_audit.sh > run_audit.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
export WORK=${WORK:-/tmp/nrt204/homology_audit_work}
RESULTS=${RESULTS:-/tmp/nrt204/FragmentPredictor/data/processed_data/homology_audit}
PY=${PY:-/home/nrt204/.conda/envs/gene_prediction_env/bin/python}
mkdir -p "$WORK/logs" "$RESULTS"
run() { echo "$(date +%T) $1"; "${@:2}" > "$WORK/logs/$1.log" 2>&1; }

run 00_stage   bash 00_stage_data.sh
# protein level
run 01_extract $PY 01_extract_fragments.py --work "$WORK"
run 02_mmseqs  bash 02_run_mmseqs.sh
run 03_bins    $PY 03_assign_bins.py --work "$WORK"
run 04_eval    $PY 04_evaluate_predictions.py --work "$WORK" --results "$RESULTS"
# nucleotide level
run 05_extract $PY 05_extract_reads_nt.py --work "$WORK"
run 06_mmseqs  bash 06_search_reads_nt.sh
run 07_assign  $PY 07_assign_reads_nt.py --work "$WORK" --results "$RESULTS"
run 08_eval    $PY 08_evaluate_reads_nt.py --work "$WORK" --results "$RESULTS"
run 09_rna     $PY 09_annotate_reads_rna.py --work "$WORK" --results "$RESULTS"
run 10_cover   $PY 10_rna_gene_coverage.py --results "$RESULTS"
echo "DONE - run homology_audit_results.ipynb"
