#!/bin/bash
# Step 0: copy the inputs from the ERDA mount to local disk ($WORK/staged).
# The mount is slow per file but fast in parallel, so files are copied with xargs -P. Already copied files are skipped.
#
# Copies: error-free train/val read CSVs (the training data of DeepCDS N), the error-free 300 bp test read CSVs,
# test labels (testset_dict.pkl, read_names_list.pkl), the predictions of DeepCDS N, MetaProdigal and FGS (Complete)
# on that test set, partition lists and genome info.
set -euo pipefail

SRC=/tmp/nrt204/FragmentPredictor/data/processed_data
WORK=${WORK:-/tmp/nrt204/homology_audit_work}
STAGE=$WORK/staged
NJOBS=${NJOBS:-32}
D=without_errors_300bp
DEEPCDS_RUN=full_model_all_genomes_seed_42_trained_final_8M_no_dropout
PREDS=$SRC/predictions/processed_predictions

mkdir -p "$STAGE/genome_partitions" "$STAGE/dataset_information"
cp -u "$SRC"/genome_partitions/{train,val,test}_partition_accessions.txt "$STAGE/genome_partitions/"
cp -u "$SRC"/dataset_information/{genomes_info_with_partitions.csv,table4_accessions} "$STAGE/dataset_information/"

# awk 1 adds the trailing newline missing in the partition files
TRAIN_VAL=$(awk 1 "$STAGE"/genome_partitions/{train,val}_partition_accessions.txt)
TEST=$(awk 1 "$STAGE"/genome_partitions/test_partition_accessions.txt)

LIST=$WORK/stage_file_list.tsv   # source <TAB> destination
: > "$LIST"
for acc in $TRAIN_VAL; do
    printf "%s\t%s\n" "$SRC/reads_processed/train_val/without_errors/csv/$acc.csv.gz" "$STAGE/train_val_csv/$acc.csv.gz" >> "$LIST"
done
for acc in $TEST; do
    printf "%s\t%s\n" "$SRC/reads_processed/test/$D/csv/$acc.csv.gz" "$STAGE/test_csv/$acc.csv.gz" >> "$LIST"
    printf "%s\t%s\n" "$SRC/testset_processed/$D/$acc/testset_dict.pkl" "$STAGE/testset/$acc/testset_dict.pkl" >> "$LIST"
    printf "%s\t%s\n" "$SRC/testset_processed/$D/$acc/read_names_list.pkl" "$STAGE/testset/$acc/read_names_list.pkl" >> "$LIST"
    printf "%s\t%s\n" "$PREDS/DeepCDS/model_without_errors/$D/$DEEPCDS_RUN/$acc/model_preds_dict.pkl" "$STAGE/preds/deepcds_n/$acc.pkl" >> "$LIST"
    printf "%s\t%s\n" "$PREDS/prodigal_preds/$D/$acc/model_preds_dict.pkl" "$STAGE/preds/metaprodigal/$acc.pkl" >> "$LIST"
    printf "%s\t%s\n" "$PREDS/fgs_preds/$D/$acc.pkl" "$STAGE/preds/fgs_complete/$acc.pkl" >> "$LIST"
done
echo "Files to stage: $(wc -l < "$LIST")"

copy_one() {
    [ -s "$2" ] && return 0
    mkdir -p "$(dirname "$2")"
    if cp "$1" "$2.part"; then mv "$2.part" "$2"; else echo "FAILED: $1" >&2; rm -f "$2.part"; return 1; fi
}
export -f copy_one
shuf "$LIST" | xargs -P "$NJOBS" -L 1 bash -c 'copy_one "$0" "$1"'   # shuffled: mixes large CSVs and small pickles

echo "Staged files: $(find "$STAGE" -type f ! -name '*.part' | wc -l)"
echo "STEP 0 DONE"
