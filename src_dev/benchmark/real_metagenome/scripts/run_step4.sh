#!/bin/bash
# Step 4 on the full sample, once the DeepCDS S+I run has finished (its timing log gets a data row at the end).
# Reads the DeepCDS GFFs directly from the ERDA mount.
set -euo pipefail
cd "$(dirname "$0")/.."
DC=/tmp/nrt204/FragmentPredictor/real_metagenome_poc/predictions/DeepCDS
PY=~/.conda/envs/poc_env/bin/python
until [ "$(wc -l < $DC/real_metagenome_poc_timing_log_indel_substitution_reads_all.tsv)" -ge 2 ]; do sleep 120; done
echo "S+I finished: $(tail -1 $DC/real_metagenome_poc_timing_log_indel_substitution_reads_all.tsv)"
( time $PY scripts/07_detection_table.py --deepcds_dir $DC ) > /tmp/nrt204/FragmentPredictor/real_metagenome_poc/logs/07_detection_table.log 2>&1
( time $PY scripts/08_analysis.py ) > /tmp/nrt204/FragmentPredictor/real_metagenome_poc/logs/08_analysis.log 2>&1
( time $PY scripts/09_spot_checks.py --deepcds_dir $DC ) > /tmp/nrt204/FragmentPredictor/real_metagenome_poc/logs/09_spot_checks.log 2>&1
echo "step 4 done"
