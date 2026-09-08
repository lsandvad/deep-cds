#!/bin/bash
#
# Run all CAMI metagenome postprocessing: the testset builder plus all three prediction
# postprocessors (Prodigal, FragGeneScanRs, DeepCDS), for all 4 EXCLUDE_UNCERTAIN_REGION_READS
# x EXCLUDE_NO_CDS_CONTIG_READS toggle combinations, so every variant a notebook might ask
# for (CAMI_metagenome, CAMI_metagenome_no_ab_initio, CAMI_metagenome_no_cds_contigs,
# CAMI_metagenome_no_ab_initio_no_cds_contigs) exists after one run. Each of the 4 scripts
# now takes --exclude-uncertain-region-reads / --exclude-no-cds-contig-reads flags that
# override the module-level toggles for that invocation only - the .py files' own defaults
# (both False) are untouched, so running any script directly still behaves as documented.
# Each script discovers whichever CAMI samples/prediction variants are actually present
# under /tmp/nrt204/FragmentPredictor, so nothing sample-specific needs to be passed here.
#
# The two *_no_cds_contigs variants additionally require cami_contigs_without_cds.py to
# have already been run for the sample(s) in question (see test_CAMI/cami_contigs_without_cds.py)
# - that step is NOT part of this script, since it hits the NCBI API and is a one-time,
# long-running fetch. Those two combinations are skipped (with a warning) if it hasn't been.
#
# Postprocessing itself is cheap relative to the original predictions (filtering
# already-computed TSVs/GFFs, not re-running FGS/Prodigal/DeepCDS), so sweeping all 4
# combinations costs minutes, not hours.
#
# Usage: ./postprocess_all_metagenome.sh [SAMPLE ...]
#   SAMPLE args are only used to check whether cami_contigs_without_cds.py has been run
#   for the no_cds_contigs variants; default: sample_0.
#
# Configured for SCARB cluster.

set -e  # Exit if any command fails

module load miniconda/24.5.0

# Initialize conda and source bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gene_prediction_env

# Resolve script paths relative to this file's own location, so this can be run from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PREPROCESSING_DIR="$SCRIPT_DIR/../../data_preprocessing"
PROJECT_ROOT="/tmp/nrt204/FragmentPredictor"

LOG_DIR="$SCRIPT_DIR/postprocess_logs"
mkdir -p "$LOG_DIR"

SAMPLES=("$@")
if [ ${#SAMPLES[@]} -eq 0 ]; then
    SAMPLES=("sample_0")
fi

# Check whether cami_contigs_without_cds.py has been run for every sample, so the
# no_cds_contigs combinations can be skipped cleanly (with a warning) instead of every
# script in that sweep failing with a FileNotFoundError.
NO_CDS_CONTIGS_READY=true
for sample in "${SAMPLES[@]}"; do
    if [ ! -f "$PROJECT_ROOT/data/processed_data/CAMI_metagenome_no_cds_contigs/${sample}_contigs_without_cds.txt" ]; then
        NO_CDS_CONTIGS_READY=false
        break
    fi
done

SCRIPTS=(
    "testset|$DATA_PREPROCESSING_DIR/postprocess_metagenome_testset.py"
    "prodigal|$SCRIPT_DIR/postprocess_prodigal_preds.py"
    "fgs|$SCRIPT_DIR/postprocess_fgs_preds.py"
    "deepcds|$SCRIPT_DIR/postprocess_model_predictions.py"
)

# variant name | --exclude-uncertain-region-reads flag | --exclude-no-cds-contig-reads flag
VARIANTS=(
    "CAMI_metagenome||"
    "CAMI_metagenome_no_ab_initio|--exclude-uncertain-region-reads|"
    "CAMI_metagenome_no_cds_contigs||--exclude-no-cds-contig-reads"
    "CAMI_metagenome_no_ab_initio_no_cds_contigs|--exclude-uncertain-region-reads|--exclude-no-cds-contig-reads"
)

echo "=============================================="
echo "Running all CAMI metagenome postprocessing"
echo "Steps: testset, prodigal, fgs, deepcds"
echo "Variants: ${#VARIANTS[@]} toggle combinations"
if [ "$NO_CDS_CONTIGS_READY" = false ]; then
    echo "[warning] cami_contigs_without_cds.py output not found for: ${SAMPLES[*]}"
    echo "          -> *_no_cds_contigs variants will be skipped. Run"
    echo "             test_CAMI/cami_contigs_without_cds.py --sample <sample> first to enable them."
fi
echo "=============================================="
echo ""

START_TIME=$(date +%s)
CURRENT=0
TOTAL_RUNS=$(( ${#SCRIPTS[@]} * ${#VARIANTS[@]} ))

for variant in "${VARIANTS[@]}"; do
    IFS='|' read -r VARIANT_NAME AB_INITIO_FLAG NO_CDS_CONTIGS_FLAG <<< "$variant"

    if [ -n "$NO_CDS_CONTIGS_FLAG" ] && [ "$NO_CDS_CONTIGS_READY" = false ]; then
        echo "[skip] $VARIANT_NAME - cami_contigs_without_cds.py hasn't been run yet"
        echo ""
        CURRENT=$((CURRENT + ${#SCRIPTS[@]}))
        continue
    fi

    echo "################################################"
    echo "# Variant: $VARIANT_NAME"
    echo "################################################"

    for step in "${SCRIPTS[@]}"; do
        CURRENT=$((CURRENT + 1))
        NAME="${step%%|*}"
        SCRIPT_PATH="${step##*|}"

        echo "=============================================="
        echo "[$CURRENT/$TOTAL_RUNS] Running: $NAME ($VARIANT_NAME)"
        echo "=============================================="

        LOG_FILE="$LOG_DIR/postprocess_${NAME}_${VARIANT_NAME}.log"
        python "$SCRIPT_PATH" $AB_INITIO_FLAG $NO_CDS_CONTIGS_FLAG 2>&1 | tee "$LOG_FILE"

        echo ""
        echo "Completed: $NAME ($VARIANT_NAME)"
        echo "Log saved to: $LOG_FILE"
        echo ""
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "=============================================="
echo "ALL METAGENOME POSTPROCESSING COMPLETE"
echo "Runs completed: $CURRENT / $TOTAL_RUNS"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Logs saved in: $LOG_DIR/"
echo "=============================================="
