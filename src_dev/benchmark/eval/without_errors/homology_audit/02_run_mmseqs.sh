#!/bin/bash
# Step 2: protein-level MMseqs2 search of the test CDS fragments (query) against the train/val CDS fragments (target).
#
#   -s 7.5 -e 10 --min-aln-len 20   maximum sensitivity; permissive E-value, so that any similarity the model could
#                                   exploit is found (E <= 1e-3 is applied later in the notebook)
#   -a                              keep the alignment backtrace, so that fident comes from the actual alignment
# The 10 best hits per query are written out; MMseqs2 sorts each query's hits by E-value, which for a fixed query
# is monotone in bitscore, so the best-bitscore hit is among them (checked in step 3).
set -euo pipefail

WORK=${WORK:-/tmp/nrt204/homology_audit_work}
MMSEQS=${MMSEQS:-/home/nrt204/.conda/envs/mmseqs_env/bin/mmseqs}
THREADS=${THREADS:-$(nproc)}
FRAG=$WORK/fragments
DB=$WORK/mmseqs_e10
mkdir -p "$DB"
echo "MMseqs2 $($MMSEQS version), $THREADS threads"

[ -s "$DB/query_db" ] || $MMSEQS createdb "$FRAG/query.fasta" "$DB/query_db" --dbtype 1
[ -s "$DB/target_db" ] || $MMSEQS createdb "$FRAG/target.fasta" "$DB/target_db" --dbtype 1
rm -rf "$DB/result_db"* "$DB/tmp"
$MMSEQS search "$DB/query_db" "$DB/target_db" "$DB/result_db" "$DB/tmp" \
    -s 7.5 -e 10 --min-aln-len 20 -a --threads "$THREADS" --compressed 1
$MMSEQS filterdb "$DB/result_db" "$DB/result_top10_db" --extract-lines 10 --threads "$THREADS"
$MMSEQS convertalis "$DB/query_db" "$DB/target_db" "$DB/result_top10_db" "$DB/result.m8" \
    --format-output "query,target,fident,alnlen,qlen,tlen,evalue,bits,qstart,qend" --threads "$THREADS"
gzip -f "$DB/result.m8"
echo "STEP 2 DONE"
