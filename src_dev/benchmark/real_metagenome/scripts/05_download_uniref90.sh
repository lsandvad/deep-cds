#!/bin/bash
# UniRef90 (UniProt release 2026_03) -> DIAMOND database, used by steps 6, 7 and 8.
# The md5 is checked against UniProt's RELEASE.metalink (2026_03: 34a83a402720e61b713e95db93d0f956).
# Usage: bash 05_download_uniref90.sh [OUT]   (default /dev/shm/nrt204_uniref90; database: OUT/uniref90.dmnd)
set -euo pipefail
OUT=${1:-/dev/shm/nrt204_uniref90}
URL=https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref90
DIAMOND=~/.conda/envs/poc_env/bin/diamond
mkdir -p "$OUT"
cd "$OUT"
curl -s "$URL/RELEASE.metalink" | grep -m1 "<version>"
MD5=$(curl -s "$URL/RELEASE.metalink" | grep -A6 'name="uniref90.fasta.gz"' | grep -o '[0-9a-f]\{32\}')
echo "expected md5 $MD5"
curl -sS --retry 5 -o uniref90.fasta.gz "$URL/uniref90.fasta.gz"
echo "$MD5  uniref90.fasta.gz" | md5sum -c
$DIAMOND version
$DIAMOND makedb --in uniref90.fasta.gz -d uniref90 --threads 64
rm uniref90.fasta.gz
