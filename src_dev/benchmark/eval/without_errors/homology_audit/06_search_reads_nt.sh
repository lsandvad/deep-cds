#!/bin/bash
# Step 6: nucleotide-level MMseqs2 search of the test reads (query) against the train/val reads (target).
#
#   --search-type 3 --strand 1   nucleotide search on the given strand only: both strands were simulated as reads of
#                                their own, so each orientation is already present in the training and test data
#   -s 7.5 -e 10 --min-aln-len 60   as in the protein-level search (60 nt = 20 aa)
#   -a                              alignment backtrace, so that fident comes from the actual alignment
# The 20 best hits per query are written out (best hit = max bitscore, selected in step 7).
set -euo pipefail

WORK=${WORK:-/tmp/nrt204/homology_audit_work}
MMSEQS=${MMSEQS:-/home/nrt204/.conda/envs/mmseqs_env/bin/mmseqs}
THREADS=${THREADS:-$(nproc)}
NT=$WORK/nt
OUT=$NT/search
TMP=${TMP:-/dev/shm/nrt204_mmseqs_nt_tmp}   # temporary search files and the uncompressed hit table
M8DIR=${M8DIR:-/dev/shm/nrt204_nt_results}  # do not fit on /tmp, so they are kept in RAM
mkdir -p "$OUT" "$M8DIR"
echo "MMseqs2 $($MMSEQS version), $THREADS threads"

[ -s "$OUT/query_db" ] || $MMSEQS createdb "$NT/query.fna" "$OUT/query_db" --dbtype 2
[ -s "$NT/target_db" ] || $MMSEQS createdb "$NT/target.fna" "$NT/target_db" --dbtype 2
rm -rf "$OUT/result_db"* "$TMP"
$MMSEQS search "$OUT/query_db" "$NT/target_db" "$OUT/result_db" "$TMP" \
    --search-type 3 --strand 1 -s 7.5 -e 10 --min-aln-len 60 -a --threads "$THREADS" --compressed 1
$MMSEQS filterdb "$OUT/result_db" "$OUT/result_top20_db" --extract-lines 20 --threads "$THREADS"
$MMSEQS convertalis "$OUT/query_db" "$NT/target_db" "$OUT/result_top20_db" "$M8DIR/result.m8" \
    --search-type 3 --format-output "query,target,fident,alnlen,qstart,qend,qlen,evalue,bits" --threads "$THREADS"
pigz -p "$THREADS" -f "$M8DIR/result.m8"
mv "$M8DIR/result.m8.gz" "$OUT/result.m8.gz"
rm -rf "$TMP"
echo "STEP 6 DONE"
