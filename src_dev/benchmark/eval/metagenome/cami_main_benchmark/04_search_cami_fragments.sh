#!/bin/bash
# Step 4: MMseqs2 search of the CAMI CDS fragments against the DeepCDS train/val fragments.
#
# Target DB = the one built by the homology audit (../../without_errors/homology_audit, steps 1-2: all labeled CDS fragments
# >= 20 aa in the error-free train/val reads, 9.75M unique proteins), reused so both analyses compare
# against exactly the same training sequences. Search settings identical to homology_audit/02_run_mmseqs.sh:
#   -s 7.5 -e 10 --min-aln-len 20 -a     (E <= 10: any similarity counts, homologous or chance)
#
# Usage: bash 04_search_cami_fragments.sh   (run inside screen)
set -euo pipefail
WORK=${WORK:-/tmp/nrt204/cami_overlap_work}
TARGET_DB=${TARGET_DB:-/tmp/nrt204/homology_audit_work/mmseqs_e10/target_db}
MMSEQS=${MMSEQS:-/home/nrt204/.conda/envs/mmseqs_env/bin/mmseqs}
THREADS=${THREADS:-$(nproc)}
D=$WORK/mmseqs
mkdir -p "$D"

[ -s "$TARGET_DB" ] || { echo "target DB not found: $TARGET_DB (run homology_audit steps 1-2)"; exit 1; }
echo "MMseqs2 version: $($MMSEQS version)"; date
[ -s "$D/query_db" ] || $MMSEQS createdb "$WORK/fragments/query.fasta" "$D/query_db" --dbtype 1
rm -rf "$D/result_db"* "$D/tmp"
$MMSEQS search "$D/query_db" "$TARGET_DB" "$D/result_db" "$D/tmp" \
    -s 7.5 -e 10 --min-aln-len 20 -a --threads "$THREADS" --compressed 1
# Top 10 hits per query (ordered by E-value = by bitscore for a fixed query); best = max bitscore in step 5
$MMSEQS filterdb "$D/result_db" "$D/result_top10_db" --extract-lines 10 --threads "$THREADS"
$MMSEQS convertalis "$D/query_db" "$TARGET_DB" "$D/result_top10_db" "$D/result.m8" \
    --format-output "query,target,fident,alnlen,qlen,tlen,evalue,bits" --threads "$THREADS"
gzip -f "$D/result.m8"
rm -rf "$D/tmp"
date; echo "STEP 4 DONE"
