"""
Which contigs - and, aggregated up, which genomes - referenced by a CAMI sample's reads
have NO CDS features in their NCBI GFF3 annotation at all. This is different from a
contig whose CDS entries all get filtered out during labeling (pseudo/partial/programmed
frameshift - see CDS_EXCLUDE_MARKERS in process_and_map_CAMI_testset.py) or one whose
reads simply don't happen to overlap the CDS(s) it does have: this checks the raw GFF3
content itself for whether NCBI has annotated any coding sequence on that record at all.

Reuses process_and_map_CAMI_testset.py's own GFF3 fetch/cache machinery (fetch_gff,
RateLimiter, extract_contig_accessions) directly, so this shares its on-disk cache and
NCBI rate limiting instead of re-implementing either.

This is a genuinely long-running script if the cache is cold: NCBI's unauthenticated
rate limit is 3 req/sec, and a CAMI sample can reference several thousand distinct
contigs (sample_0: 6,327 - about 35 minutes at 3 req/sec, ~10-11 minutes with an
--api-key). Use --limit while testing, and consider --skip-fetch if you already have a
populated gff_cache/ from a labeling run and just want to check what's there.

Output is written under <project-root>/data/processed_data/CAMI_metagenome_no_cds_contigs/
(not the cwd), since this is meant to be kept for later assessment - the postprocess_metagenome
pipeline's EXCLUDE_NO_CDS_CONTIG_READS toggle (see postprocess_metagenome_testset.py and the
three postprocess_metagenome/postprocess_*_preds.py scripts) and src_dev/benchmark/eval/metagenome/cami_sample_statistics/cami_no_cds_contig_predictions.py
both read the *_contigs_without_cds.txt file back from there.

Usage:
    python3 cami_contigs_without_cds.py --sample sample_0
    python3 cami_contigs_without_cds.py --sample sample_0 --skip-fetch    # cache-only, no network
    python3 cami_contigs_without_cds.py --sample sample_0 --limit 200     # smoke test
"""
import argparse
import gzip
import re
import sys
from pathlib import Path

# Reuse process_and_map_CAMI_testset.py's own GFF3 fetch/cache/rate-limiting machinery
# rather than re-implementing any of it, so this shares its on-disk cache exactly.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "data_preprocessing"))
from process_and_map_CAMI_testset import (  # noqa: E402
    API_KEY_RATE,
    NO_KEY_RATE,
    RateLimiter,
    fetch_gff,
)

SCARB_ROOT = Path("/tmp/nrt204/FragmentPredictor")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--sample", default="sample_0", help="CAMI sample name (default: sample_0)")
parser.add_argument("--data-root", default=str(SCARB_ROOT / "data" / "raw_data" / "CAMI_datasets"),
                     help="Raw CAMI data directory (default: matches process_and_map_CAMI_testset.py)")
parser.add_argument("--gff-cache-dir", default=None,
                     help="Override: GFF3 cache directory (default: <data-root>/gff_cache, "
                          "shared with the main labeling script)")
parser.add_argument("--out-dir", default=str(SCARB_ROOT / "data" / "processed_data" / "CAMI_metagenome_no_cds_contigs"),
                     help="Directory the *_contigs_without_cds.txt / *_genomes_without_cds.txt lists are "
                          "written to (persisted for later reuse, not the cwd)")
parser.add_argument("--skip-fetch", action="store_true",
                     help="Never hit the network; only check whatever GFF3s are already cached")
parser.add_argument("--api-key", default=None, help="NCBI API key (raises rate limit from 3 to 10 req/sec)")
parser.add_argument("--rate", type=float, default=None, help="Override request rate (req/sec)")
parser.add_argument("--limit", type=int, default=None, help="Only check the first N contigs (for a quick smoke test)")


def build_contig_genome_map(reads_mapping_gz):
    """One pass over reads_mapping.tsv.gz: distinct contig accession -> set of genome_id(s)
    that reference it (normally exactly one genome per contig - stored as a set so any
    surprise there is visible rather than silently overwritten).

    read_id format: "[<plasmid_copy_index>]<accession>-<pos>/<mate>" - the same
    convention process_and_map_CAMI_testset.py's extract_contig_accessions() parses.
    """
    pattern = re.compile(r"^\d*(.+)-\d+/[12]$")
    contig_to_genomes = {}
    with gzip.open(reads_mapping_gz, "rt") as fh:
        header = next(fh).rstrip("\n").split("\t")
        genome_col = header.index("genome_id")
        read_id_col = header.index("read_id")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(genome_col, read_id_col):
                continue
            m = pattern.match(parts[read_id_col])
            if not m:
                continue
            contig_to_genomes.setdefault(m.group(1), set()).add(parts[genome_col])
    return contig_to_genomes


def has_any_cds(gff_path):
    """Whether a GFF3 file contains at least one raw CDS feature row - a direct read of
    the file's own content, deliberately NOT using parse_gff3_cds()'s quality-filtered
    output (that function is for building the labeling truth set; this is asking the
    more literal "does NCBI's annotation for this contig include any CDS at all")."""
    with open(gff_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 3 and cols[2] == "CDS":
                return True
    return False


def main():
    args = parser.parse_args()

    reads_mapping_path = Path(args.data_root) / f"{args.sample}_reads" / "reads_mapping.tsv.gz"
    gff_cache_dir = Path(args.gff_cache_dir) if args.gff_cache_dir else Path(args.data_root) / "gff_cache"
    gff_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building contig -> genome map from {reads_mapping_path} ...", flush=True)
    contig_to_genomes = build_contig_genome_map(reads_mapping_path)
    contigs = sorted(contig_to_genomes.keys())
    if args.limit:
        contigs = contigs[:args.limit]
    print(f"{len(contigs)} distinct contigs to check"
          + (f" (limited from {len(contig_to_genomes)})" if args.limit else "") + ".", flush=True)

    multi_genome_contigs = {c: g for c, g in contig_to_genomes.items() if len(g) > 1}
    if multi_genome_contigs:
        print(f"[note] {len(multi_genome_contigs)} contig(s) are referenced by more than one genome_id - "
              f"unexpected, worth a look: {list(multi_genome_contigs.items())[:5]}", flush=True)

    rate = args.rate or (API_KEY_RATE if args.api_key else NO_KEY_RATE)
    rate_limiter = RateLimiter(rate)
    print(f"GFF3 cache: {gff_cache_dir}"
          + (" (cache-only, --skip-fetch)" if args.skip_fetch else f" (fetching missing at up to {rate:.0f} req/sec)"),
          flush=True)

    contigs_without_cds = []
    contigs_not_available = []  # not cached, and either --skip-fetch or the fetch failed

    for i, contig in enumerate(contigs, 1):
        cache_path = gff_cache_dir / f"{contig}.gff3"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            gff_path = cache_path
        elif args.skip_fetch:
            gff_path = None
        else:
            gff_path = fetch_gff(contig, gff_cache_dir, rate_limiter, args.api_key)

        if gff_path is None:
            contigs_not_available.append(contig)
        elif not has_any_cds(gff_path):
            contigs_without_cds.append(contig)

        if i % 200 == 0 or i == len(contigs):
            print(f"  ... {i}/{len(contigs)} contigs checked "
                  f"({len(contigs_without_cds)} with no CDS, {len(contigs_not_available)} unavailable)", flush=True)

    # Aggregate to genome level: a genome counts as "no CDS annotations at all" only if
    # EVERY one of its contigs that we could actually check has zero CDS - contigs we
    # couldn't fetch/check are excluded from the determination rather than assumed either way.
    checked_contigs = set(contigs) - set(contigs_not_available)
    no_cds_set = set(contigs_without_cds)

    genome_contig_status = {}  # genome_id -> (n_contigs_checked, n_contigs_without_cds)
    for contig in checked_contigs:
        for genome in contig_to_genomes[contig]:
            n_checked, n_no_cds = genome_contig_status.get(genome, (0, 0))
            genome_contig_status[genome] = (
                n_checked + 1,
                n_no_cds + (1 if contig in no_cds_set else 0),
            )

    genomes_fully_without_cds = [
        g for g, (n_checked, n_no_cds) in genome_contig_status.items()
        if n_checked > 0 and n_checked == n_no_cds
    ]

    print(f"\n=== Contigs/genomes with zero CDS annotations in their GFF3 ({args.sample}) ===")
    print(f"Contigs checked:                        {len(checked_contigs):,} / {len(contigs):,}")
    if contigs_not_available:
        print(f"Contigs not available:                  {len(contigs_not_available):,} (not cached"
              + ("" if args.skip_fetch else " and fetch failed") + " - excluded from the counts below)")
    print(f"Contigs with NO CDS at all:              {len(contigs_without_cds):,}")
    print(f"\nGenomes checked (>=1 contig checked):    {len(genome_contig_status):,}")
    print(f"Genomes where EVERY contig has no CDS:   {len(genomes_fully_without_cds):,}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always written (even if empty) so downstream scripts that load this file back
    # (postprocess_metagenome_testset.py's EXCLUDE_NO_CDS_CONTIG_READS, etc.) can tell
    # "checked, none found" apart from "never computed for this sample".
    contigs_out_path = out_dir / f"{args.sample}_contigs_without_cds.txt"
    contigs_out_path.write_text("\n".join(sorted(contigs_without_cds)) + ("\n" if contigs_without_cds else ""))
    print(f"\nFull list of no-CDS contigs written to {contigs_out_path}")

    genomes_out_path = out_dir / f"{args.sample}_genomes_without_cds.txt"
    genomes_out_path.write_text("\n".join(sorted(genomes_fully_without_cds)) + ("\n" if genomes_fully_without_cds else ""))
    print(f"Full list of no-CDS genomes written to {genomes_out_path}")

    if contigs_not_available:
        unavailable_out_path = out_dir / f"{args.sample}_contigs_not_available.txt"
        unavailable_out_path.write_text("\n".join(sorted(contigs_not_available)) + "\n")
        print(f"Contigs that couldn't be checked (not cached"
              + ("" if args.skip_fetch else " and fetch failed") + f") written to {unavailable_out_path}")


if __name__ == "__main__":
    main()
