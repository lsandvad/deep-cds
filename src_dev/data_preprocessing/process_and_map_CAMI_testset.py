#!/usr/bin/env python3
"""
Label CAMI-simulated reads with ground-truth per-codon CDS reading-frame status.

Reuses the exact RF0/RF1/RF2 labeling + codon-completeness-trimming functions from
src_dev/data_preprocessing/process_reads_with_indels.py, so output matches the rest
of the benchmarking pipeline's convention. What's different here is only the data
ingestion, since CAMI reads don't come from the Mason-based pipeline that code was
built around:

  - Contig accessions are taken from the sample's reads_mapping.tsv.gz (only contigs
    actually covered by reads are fetched, not every contig in every source genome).
  - Ground-truth GFF3 annotations for those contigs are fetched directly from NCBI
    (cached on disk under --gff-cache-dir, shared/reused across samples and re-runs).
  - Alignments are read straight from the sample's BAM files (real CIGAR/FLAG),
    instead of Mason's per-strand simulated-reads BAMs.

Known simplifications vs. the original pipeline (both are QC/verification gates that
don't affect the RF/codon labels themselves, which depend only on CIGAR + CDS coords):
  - No protein.faa-based proteome verification of translated coding fragments (we
    don't have local protein FASTA files for these accessions).
  - No MD:Z-based indel/substitution bookkeeping for check_cds_quality's frame-shift
    direction check (would require reversing MD:Z tags for minus-strand reads, which
    is fragile and not needed for the labels themselves).

Re-run this per sample by pointing --bam-dir / --reads-mapping at a new sample; the
GFF cache is shared and only grows across runs.
"""

import argparse
import csv
import gzip
import http.client
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import pysam

# SCARB cluster data root - all of this repo's benchmark/postprocess scripts read and
# write large data under here rather than the (networked, slower) repo checkout itself.
SCARB_ROOT = Path("/tmp/nrt204/FragmentPredictor")

from process_reads_with_indels import (  # noqa: E402
    assign_codon_labels,
    check_cds_quality,
    generate_rf_labels_with_indels,
    get_position_gene_overlaps,
    mark_intervals,
    parse_cigar,
    quality_check_cds_annotations,
)

NCBI_SVIEWER_URL = "https://www.ncbi.nlm.nih.gov/sviewer/viewer.fcgi"
NCBI_TAXONOMY_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2/taxonomy/taxon"
NO_KEY_RATE = 3.0  # requests/sec, NCBI's unauthenticated rate limit
API_KEY_RATE = 10.0  # requests/sec, with an NCBI API key

# Stable, well-known root tax IDs for the domains (superkingdoms) - these never change,
# so a lineage containing one of them unambiguously identifies the read's domain.
DOMAIN_ROOT_TAXIDS = {2: "Bacteria", 2157: "Archaea", 2759: "Eukaryota", 10239: "Viruses"}

# CDS quality/confidence markers, split into two tiers:
#   - EXCLUDE: annotation is dropped entirely, and any read overlapping it is dropped too
#     (matches process_reads_with_indels.py's read_is_in_uncertain_range behavior).
#   - TAG: annotation is kept and labeled as CDS normally, but reads overlapping it are
#     flagged via the uncertain_region_overlap output column so they can be filtered
#     in/out downstream without re-running the pipeline.
CDS_EXCLUDE_MARKERS = (
    "pseudo=true",
    "partial=true",
    "note=programmed frameshift",
)
CDS_TAG_MARKERS = (
    "product=hypothetical protein",
    "ab initio prediction",
)

COMPLEMENT_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT_TABLE)[::-1]


def normalize_cigar(cigar: str) -> str:
    """Convert extended CIGAR ops (=, X) to legacy M; positionally identical for our purposes
    since both consume one query and one reference base per unit, same as M."""
    return cigar.replace("=", "M").replace("X", "M")


def reverse_cigar(cigar: str) -> str:
    ops = parse_cigar(cigar)
    return "".join(f"{length}{op}" for op, length in reversed(ops))


class RateLimiter:
    def __init__(self, per_second: float):
        self.min_interval = 1.0 / per_second
        self._last = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


def extract_contig_accessions(reads_mapping_gz: Path) -> set:
    """Distinct contig accessions actually referenced by reads.

    read_id format: "[<plasmid_copy_index>]<accession>-<pos>/<mate>", e.g.
    "NZ_JBDQBI010000009.1-9644/1" or "3NZ_CP029594.1-623/1" (plasmid copy index 3).
    """
    accessions = set()
    pattern = re.compile(r"^\d*(.+)-\d+/[12]$")
    with gzip.open(reads_mapping_gz, "rt") as fh:
        next(fh)  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            m = pattern.match(parts[3])
            if m:
                accessions.add(m.group(1))
    return accessions


def fetch_gff(accession: str, cache_dir: Path, rate_limiter: RateLimiter, api_key: str = None) -> Path:
    """Fetch (or reuse cached) ground-truth GFF3 for a single NCBI nucleotide accession."""
    cache_path = cache_dir / f"{accession}.gff3"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    url = f"{NCBI_SVIEWER_URL}?id={accession}&db=nuccore&report=gff3&retmode=text"
    if api_key:
        url += f"&api_key={api_key}"

    for attempt in range(5):
        rate_limiter.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FragmentPredictor-cds-labeling"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if not data.startswith(b"##gff-version"):
                raise ValueError(f"unexpected response body (first 100B): {data[:100]!r}")
            cache_path.write_bytes(data)
            return cache_path
        except (urllib.error.HTTPError, urllib.error.URLError, http.client.HTTPException, ValueError, TimeoutError, ConnectionError) as exc:
            wait = 2**attempt
            print(f"  [retry {attempt + 1}/5] {accession}: {exc} (waiting {wait}s)", file=sys.stderr)
            time.sleep(wait)

    print(f"  [FAILED] could not fetch GFF3 for {accession} after 5 attempts", file=sys.stderr)
    return None


def fetch_accession_taxid(accession: str, cache_dir: Path, rate_limiter: RateLimiter, api_key: str = None):
    """Resolve a contig accession to its tax_id via NCBI's esummary endpoint - a tiny (~1KB)
    JSON response regardless of the underlying sequence's size. Deliberately independent of
    fetch_gff()/the CDS GFF3 fetch, since that can be hundreds of MB for a full chromosome
    (e.g. a human contig) - domain tagging shouldn't have to pay that cost or depend on it
    succeeding. Returns None if unresolvable."""
    cache_path = cache_dir / f"{accession}.txt"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return int(cache_path.read_text().strip())

    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=nuccore&id={accession}&retmode=json"
    if api_key:
        url += f"&api_key={api_key}"

    for attempt in range(5):
        rate_limiter.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FragmentPredictor-cds-labeling"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            uid = data["result"]["uids"][0]
            tax_id = int(data["result"][uid]["taxid"])
            cache_path.write_text(str(tax_id))
            return tax_id
        except (urllib.error.HTTPError, urllib.error.URLError, http.client.HTTPException, KeyError, IndexError, ValueError, TimeoutError, ConnectionError) as exc:
            wait = 2**attempt
            print(f"  [retry {attempt + 1}/5] esummary {accession}: {exc} (waiting {wait}s)", file=sys.stderr)
            time.sleep(wait)

    print(f"  [FAILED] could not resolve tax_id for {accession} after 5 attempts", file=sys.stderr)
    return None


def fetch_domain(tax_id: int, cache_dir: Path, rate_limiter: RateLimiter, api_key: str = None) -> str:
    """Resolve a tax ID to its domain (superkingdom: Bacteria/Archaea/Eukaryota/Viruses) via
    NCBI's taxonomy API, caching the result to disk. Keyed by tax_id (not contig accession),
    so many strains sharing a species cost one lookup, not one per contig."""
    cache_path = cache_dir / f"{tax_id}.txt"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_text().strip()

    url = f"{NCBI_TAXONOMY_URL}/{tax_id}"
    if api_key:
        url += f"?api_key={api_key}"

    for attempt in range(5):
        rate_limiter.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FragmentPredictor-cds-labeling"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            lineage = data["taxonomy_nodes"][0]["taxonomy"]["lineage"]
            domain = "Unknown"
            for ancestor in lineage:
                if ancestor in DOMAIN_ROOT_TAXIDS:
                    domain = DOMAIN_ROOT_TAXIDS[ancestor]
                    break
            cache_path.write_text(domain)
            return domain
        except (urllib.error.HTTPError, urllib.error.URLError, http.client.HTTPException, KeyError, IndexError, ValueError, TimeoutError, ConnectionError) as exc:
            wait = 2**attempt
            print(f"  [retry {attempt + 1}/5] tax_id {tax_id}: {exc} (waiting {wait}s)", file=sys.stderr)
            time.sleep(wait)

    print(f"  [FAILED] could not resolve domain for tax_id {tax_id} after 5 attempts", file=sys.stderr)
    return "Unknown"


def fetch_all_gffs(accessions, cache_dir: Path, rate: float, api_key: str = None) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rate_limiter = RateLimiter(rate)
    paths = {}
    todo = sorted(accessions)
    n_cached = sum(1 for a in todo if (cache_dir / f"{a}.gff3").exists() and (cache_dir / f"{a}.gff3").stat().st_size > 0)
    print(f"Fetching GFF3 annotations for {len(todo)} contigs ({n_cached} already cached)...")
    for i, accession in enumerate(todo, 1):
        path = fetch_gff(accession, cache_dir, rate_limiter, api_key)
        if path is not None:
            paths[accession] = path
        if i % 200 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} processed ({len(paths)} succeeded)")
    return paths


def parse_gff3_cds(gff_path: Path):
    """
    Parse a single-contig GFF3 into a dict:
        {"length": int, "+": StrandAnnotation, "-": StrandAnnotation}
    where StrandAnnotation is {"cds": {seqid: [[start,end],...]}, "tag": {seqid: [[start,end],...]}}.

    "cds" is the full set of CDS intervals used for RF labeling: normal CDS plus CDS flagged
    with a CDS_TAG_MARKERS marker (hypothetical protein / ab initio prediction / programmed
    frameshift) - these are lower-confidence but still labeled as coding. "tag" is the subset
    of "cds" that came from a flagged annotation, kept separately so reads overlapping it can
    be marked uncertain_region_overlap=True downstream without excluding them.

    Pseudogenes and CDS flagged pseudo=true/partial=true (CDS_EXCLUDE_MARKERS) are dropped
    from "cds" entirely and are not tracked at all here - process_bam separately builds an
    exclude-interval list from the raw GFF to drop reads overlapping those regions outright,
    matching process_reads_with_indels.py's read_is_in_uncertain_range behavior.

    Minus-strand coordinates are converted directly into the reverse-complement coordinate
    frame (contig length from the GFF's own ##sequence-region pragma, no local FASTA needed) -
    mirrors convert_complement_coordinates().

    Domain (superkingdom) tagging does NOT go through this function - see fetch_accession_taxid()/
    fetch_domain(), which resolve it via a separate, tiny esummary lookup keyed on the contig
    accession directly, independent of this (potentially huge, e.g. a full human chromosome) GFF3.
    """
    contig_length = None
    plus_cds, plus_tag = {}, {}
    minus_cds_raw, minus_tag_raw = {}, {}
    plus_exclude_raw, minus_exclude_raw = {}, {}

    with open(gff_path) as fh:
        for line in fh:
            if line.startswith("##sequence-region"):
                contig_length = int(line.split()[3])
                continue
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "CDS":
                continue

            seqid, start, end, strand, attrs = cols[0], int(cols[3]), int(cols[4]), cols[6], cols[8]
            attrs_lower = attrs.lower()
            coord = [start, end]

            is_pseudo = "pseudogene" in attrs_lower or any(marker in attrs_lower for marker in CDS_EXCLUDE_MARKERS)
            is_tagged = any(marker in attrs_lower for marker in CDS_TAG_MARKERS)

            if strand == "+":
                if is_pseudo:
                    plus_exclude_raw.setdefault(seqid, []).append(coord)
                    continue
                plus_cds.setdefault(seqid, []).append(coord)
                if is_tagged:
                    plus_tag.setdefault(seqid, []).append(coord)
            elif strand == "-":
                if is_pseudo:
                    minus_exclude_raw.setdefault(seqid, []).append(coord)
                    continue
                minus_cds_raw.setdefault(seqid, []).append(coord)
                if is_tagged:
                    minus_tag_raw.setdefault(seqid, []).append(coord)

    if contig_length is None:
        raise ValueError(f"no ##sequence-region pragma found in {gff_path}")

    def to_minus_view(raw):
        return {seqid: [[contig_length - end + 1, contig_length - start + 1] for start, end in coords] for seqid, coords in raw.items()}

    minus_cds = to_minus_view(minus_cds_raw)
    minus_tag = to_minus_view(minus_tag_raw)
    plus_exclude = plus_exclude_raw
    minus_exclude = to_minus_view(minus_exclude_raw)

    # Quality-check (dedup / length-validity) the labeling set only; tag/exclude interval
    # lists are used purely for boolean overlap checks and don't need this.
    # NB: quality_check_cds_annotations indexes cds_coords_uncertain[assembly] directly
    # (no .setdefault), so it needs a defaultdict here - a plain {} raises KeyError as
    # soon as any CDS fails its internal length-multiple-of-3 check.
    plus_cds, _ = quality_check_cds_annotations(plus_cds, defaultdict(list))
    minus_cds, _ = quality_check_cds_annotations(minus_cds, defaultdict(list))

    return {
        "length": contig_length,
        "+": {"cds": plus_cds, "tag": plus_tag, "exclude": plus_exclude},
        "-": {"cds": minus_cds, "tag": minus_tag, "exclude": minus_exclude},
    }


def derive_seq_errors(cigar: str) -> str:
    """Indel-only error positions (1-based, read-local, in the read's own 5'->3' orientation -
    i.e. computed from the same oriented `cigar` string used for labeling), matching the existing
    benchmarking pipeline's indel_positions/fasta seq_errors convention (mark_errors()'s I/D-only
    entries, substitutions excluded). Derived directly from the CIGAR rather than MD:Z: insertions
    and deletions are already fully specified by the CIGAR alone, so no MD:Z tag is needed - which
    matters here since we never reconstruct/reverse MD:Z for minus-strand reads (see module docstring).
    Returns "None" (matching the reference fasta convention) when the read has no indels.
    """
    errors = []
    read_pos = 0
    for op, length in parse_cigar(cigar):
        if op == "M":
            read_pos += length
        elif op == "I":
            errors.extend(f"{read_pos + i + 1}I" for i in range(length))
            read_pos += length
        elif op == "D":
            errors.extend(f"{read_pos + 1}D" for _ in range(length))
    return ",".join(errors) if errors else "None"


def label_read(seqs_len: int, cigar: str, start_coord: int, cds_overlaps: dict):
    """
    Run the read through all three reading frames (RF0/RF1/RF2), applying the same
    codon-completeness trimming as process_reads_with_indels.py: each frame's usable
    length is truncated so only whole codons survive, and generate_rf_labels_with_indels
    walks the CIGAR to keep per-base reference coordinates (hence CDS overlap) correct
    across insertions/deletions. assign_codon_labels then collapses any codon touching
    a non-coding position (including a read-edge partial codon) down to non-coding.

    Labeling is purely position-based (CIGAR + CDS coordinate overlap) - it never looks at
    nucleotide identity, so reads with N's or other ambiguity codes are labeled exactly like
    any other read; there's deliberately no amino-acid translation here (not needed for the
    evaluation pipeline, and translation is where non-ACGT bases used to blow up on).

    Returns dict with per-RF {labels_nt, labels_aa} plus the codon-level intervals from
    mark_intervals (read-local, 1-based [start, end, rf] triples).
    """
    if seqs_len % 3 == 0:
        seqs_len_rf0, seqs_len_rf1, seqs_len_rf2 = seqs_len, seqs_len - 3, seqs_len - 3
    elif seqs_len % 3 == 1:
        seqs_len_rf0, seqs_len_rf1, seqs_len_rf2 = seqs_len - 1, seqs_len - 1, seqs_len - 4
    else:  # % 3 == 2
        seqs_len_rf0, seqs_len_rf1, seqs_len_rf2 = seqs_len - 2, seqs_len - 2, seqs_len - 2

    rf_results = {}
    for rf_name, offset, rf_len in (("RF0", 0, seqs_len_rf0), ("RF1", 1, seqs_len_rf1), ("RF2", 2, seqs_len_rf2)):
        if rf_len < 3:
            rf_results[rf_name] = {"labels_nt": [], "labels_aa": []}
            continue
        labels_aa, labels_nt, _insertions = generate_rf_labels_with_indels(start_coord + offset, rf_len, cds_overlaps, cigar, rf_name)
        rf_results[rf_name] = {"labels_nt": labels_nt, "labels_aa": labels_aa}

    cds_overlaps_read, indel_cds_connect = mark_intervals(
        rf_results["RF0"]["labels_nt"], rf_results["RF1"]["labels_nt"], rf_results["RF2"]["labels_nt"],
        rf_results["RF0"]["labels_aa"], rf_results["RF1"]["labels_aa"], rf_results["RF2"]["labels_aa"],
    )
    keep = check_cds_quality(True, cds_overlaps_read, indel_cds_connect, indel_errors=None)

    return rf_results, cds_overlaps_read, indel_cds_connect, keep


def get_or_fetch_contig_cds(contig: str, cds_by_contig: dict, gff_cache_dir: Path, rate_limiter: RateLimiter, api_key: str, allow_fetch: bool, stats: dict):
    """Return parse_gff3_cds()'s dict for a contig, fetching+parsing its GFF3 on first use
    (from disk cache if present, else from NCBI) and memoizing the result in cds_by_contig."""
    if contig in cds_by_contig:
        return cds_by_contig[contig]

    gff_path = gff_cache_dir / f"{contig}.gff3"
    if not (gff_path.exists() and gff_path.stat().st_size > 0):
        if not allow_fetch:
            cds_by_contig[contig] = None
            return None
        gff_path = fetch_gff(contig, gff_cache_dir, rate_limiter, api_key)
        stats["contigs_fetched"] += 1
        if gff_path is None:
            cds_by_contig[contig] = None
            return None

    try:
        annotation = parse_gff3_cds(gff_path)
    except ValueError as exc:
        print(f"  [WARN] {contig}: {exc}", file=sys.stderr)
        cds_by_contig[contig] = None
        return None

    cds_by_contig[contig] = annotation
    return cds_by_contig[contig]


def get_or_fetch_domain_for_accession(contig: str, accession_to_taxid: dict, domain_by_taxid: dict, taxonomy_cache_dir: Path,
                                       rate_limiter: RateLimiter, api_key: str, allow_fetch: bool, stats: dict) -> str:
    """Resolve a contig accession straight to its domain (Bacteria/Archaea/Eukaryota/Viruses),
    via two independently-cached, memoized hops (accession->tax_id, tax_id->domain). Deliberately
    has nothing to do with the CDS GFF3 fetch/cache - see fetch_accession_taxid()'s docstring for
    why: a full human chromosome's GFF3 is hundreds of MB, but its tax_id lookup is ~1KB either way.
    Returns "Unknown" if either hop is unresolvable."""
    if contig not in accession_to_taxid:
        taxid_cache_dir = taxonomy_cache_dir / "accession_taxid"
        taxid_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = taxid_cache_dir / f"{contig}.txt"
        already_cached = cache_path.exists() and cache_path.stat().st_size > 0
        if not already_cached and not allow_fetch:
            accession_to_taxid[contig] = None
        else:
            accession_to_taxid[contig] = fetch_accession_taxid(contig, taxid_cache_dir, rate_limiter, api_key)
    tax_id = accession_to_taxid[contig]

    if tax_id is None:
        return "Unknown"
    if tax_id in domain_by_taxid:
        return domain_by_taxid[tax_id]

    domain_cache_dir = taxonomy_cache_dir / "taxid_domain"
    domain_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = domain_cache_dir / f"{tax_id}.txt"
    already_cached = cache_path.exists() and cache_path.stat().st_size > 0
    if not already_cached and not allow_fetch:
        domain_by_taxid[tax_id] = "Unknown"
        return "Unknown"

    domain = fetch_domain(tax_id, domain_cache_dir, rate_limiter, api_key)
    if not already_cached:
        stats["taxids_fetched"] += 1
    domain_by_taxid[tax_id] = domain
    return domain


def process_bam(bam_path: Path, genome_bin_id: str, cds_by_contig: dict, gff_cache_dir: Path,
                 accession_to_taxid: dict, domain_by_taxid: dict, taxonomy_cache_dir: Path, rate_limiter: RateLimiter,
                 api_key: str, allow_fetch: bool, writer: csv.writer, fasta_fh, stats: dict, limit: int = None):
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam:
            if limit is not None and stats["reads_seen"] >= limit:
                return
            stats["reads_seen"] += 1

            contig = read.reference_name
            # Domain resolution is independent of the CDS/GFF3 fetch below (cheap either way,
            # even for a chromosome-scale contig), so it's looked up first and unconditionally.
            domain = get_or_fetch_domain_for_accession(contig, accession_to_taxid, domain_by_taxid, taxonomy_cache_dir, rate_limiter, api_key, allow_fetch, stats)

            contig_cds = get_or_fetch_contig_cds(contig, cds_by_contig, gff_cache_dir, rate_limiter, api_key, allow_fetch, stats)
            if contig_cds is None:
                stats["skipped_no_gff"] += 1
                continue
            # Labeling is position-based only (CIGAR + CDS overlap), so N's and other ambiguity
            # codes are fine and kept as-is; this just guards against a genuinely missing SEQ
            # field (e.g. some secondary/supplementary alignments), which would crash string ops.
            if read.query_sequence is None:
                stats["skipped_ambiguous_base"] += 1
                continue

            contig_length = contig_cds["length"]
            cigar_fwd = normalize_cigar(read.cigarstring)

            if read.is_reverse:
                seq = reverse_complement(read.query_sequence)
                cigar = reverse_cigar(cigar_fwd)
                # 1-based start of the read's own 5'->3' walk, in the RC coordinate frame
                # (see parse_gff3_cds: minus-strand CDS coords already live in this frame).
                start_coord = contig_length - read.reference_end + 1
                strand_view = contig_cds["-"]
                strand = "-"
            else:
                seq = read.query_sequence
                cigar = cigar_fwd
                start_coord = read.reference_start + 1
                strand_view = contig_cds["+"]
                strand = "+"

            seq_len = len(seq)

            # Reads overlapping a pseudogene / pseudo=true / partial=true CDS are dropped
            # outright (matches process_reads_with_indels.py's read_is_in_uncertain_range).
            exclude_intervals = strand_view["exclude"].get(contig, [])
            if get_position_gene_overlaps(exclude_intervals, start_coord, seq_len):
                stats["skipped_partial_or_pseudo"] += 1
                continue

            # Reads overlapping a hypothetical-protein / ab-initio / programmed-frameshift
            # CDS are still labeled as coding, just flagged for optional downstream filtering.
            tag_intervals = strand_view["tag"].get(contig, [])
            uncertain_region_overlap = bool(get_position_gene_overlaps(tag_intervals, start_coord, seq_len))

            cds_assembly = strand_view["cds"].get(contig, [])
            cds_overlaps = get_position_gene_overlaps(cds_assembly, start_coord, seq_len)

            rf_results, cds_overlaps_read, indel_cds_connect, keep = label_read(seq_len, cigar, start_coord, cds_overlaps)
            if not keep:
                stats["skipped_quality_check"] += 1
                continue

            seq_errors = derive_seq_errors(cigar)
            # Both mates of a pair (and, rarely, both strand orientations of the same read_name)
            # share read.query_name, so it alone isn't a unique record ID - append a running
            # count, unique across the whole output, matching your requested "_seqX" suffix.
            stats["reads_written"] += 1
            seq_id = f"{read.query_name}_seq{stats['reads_written']}"

            writer.writerow([
                seq_id,
                seq,
                cds_overlaps_read,
                indel_cds_connect,
                start_coord,
                contig,
                genome_bin_id,
                cigar,
                strand,
                uncertain_region_overlap,
                domain,
                seq_errors,
                rf_results["RF0"]["labels_aa"],
                rf_results["RF1"]["labels_aa"],
                rf_results["RF2"]["labels_aa"],
            ])

            # FASTA header mirrors the existing reads_processed/**/fasta/ convention
            # (seq_id|strand|contig|cds_coords|seq_errors), with uncertain_region_overlap and
            # domain appended.
            fasta_fh.write(f">{seq_id}|{strand}|{contig}|{cds_overlaps_read}|{seq_errors}|{uncertain_region_overlap}|{domain}\n{seq}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=str(SCARB_ROOT / "data" / "raw_data" / "CAMI_datasets"), help="CAMI data directory")
    parser.add_argument("--sample", default="sample_0", help="Sample name, e.g. sample_0 (derives bam-dir/reads-mapping/out defaults)")
    parser.add_argument("--bam-dir", default=None, help="Override: directory of per-genome BAM files")
    parser.add_argument("--reads-mapping", default=None, help="Override: path to reads_mapping.tsv.gz")
    parser.add_argument("--gff-cache-dir", default=None, help="Override: shared GFF3 cache directory (default: <data-root>/gff_cache)")
    parser.add_argument("--taxonomy-cache-dir", default=None, help="Override: shared tax_id->domain cache directory (default: <data-root>/taxonomy_cache)")
    parser.add_argument("--out", default=None, help="Override: output TSV.gz path (default: <out-dir>/<sample>_cds_labels.tsv.gz)")
    parser.add_argument(
        "--out-dir",
        default=str(SCARB_ROOT / "data" / "processed_data" / "reads_processed" / "test" / "CAMI_metagenome"),
        help="Directory for the default output filename (ignored if --out is given)",
    )
    parser.add_argument("--api-key", default=None, help="NCBI API key (raises rate limit from 3 to 10 req/sec)")
    parser.add_argument("--rate", type=float, default=None, help="Override request rate (req/sec)")
    parser.add_argument("--skip-fetch", action="store_true", help="Never hit the network; only use whatever GFFs are already cached")
    parser.add_argument("--gff-only", action="store_true", help="Eagerly prefetch GFFs for every contig in reads-mapping, then exit (no read labeling). Useful to warm the cache ahead of a full run.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many reads total (smoke-testing). GFFs are fetched on demand, only for contigs actually touched, so a small --limit finishes fast.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    bam_dir = Path(args.bam_dir) if args.bam_dir else data_root / f"{args.sample}_bam"
    reads_mapping = Path(args.reads_mapping) if args.reads_mapping else data_root / f"{args.sample}_reads" / "reads_mapping.tsv.gz"
    gff_cache_dir = Path(args.gff_cache_dir) if args.gff_cache_dir else data_root / "gff_cache"
    taxonomy_cache_dir = Path(args.taxonomy_cache_dir) if args.taxonomy_cache_dir else data_root / "taxonomy_cache"
    out_path = Path(args.out) if args.out else Path(args.out_dir) / f"{args.sample}_cds_labels.tsv.gz"
    rate = args.rate or (API_KEY_RATE if args.api_key else NO_KEY_RATE)
    gff_cache_dir.mkdir(parents=True, exist_ok=True)
    taxonomy_cache_dir.mkdir(parents=True, exist_ok=True)
    rate_limiter = RateLimiter(rate)

    if args.gff_only:
        print(f"Extracting contig accessions from {reads_mapping} ...")
        accessions = extract_contig_accessions(reads_mapping)
        print(f"  {len(accessions)} distinct contigs referenced by reads")
        fetch_all_gffs(accessions, gff_cache_dir, rate, args.api_key)
        print("--gff-only set, exiting after fetch step.")
        return

    bam_files = sorted(bam_dir.glob("*.bam"))
    print(f"Labeling reads from {len(bam_files)} BAM files in {bam_dir} ...")
    print(f"GFFs are fetched on demand into {gff_cache_dir} as new contigs are encountered" + (" (network fetch disabled, --skip-fetch)" if args.skip_fetch else f" (up to {rate:.0f} req/sec)"))

    cds_by_contig = {}
    accession_to_taxid = {}
    domain_by_taxid = {}
    stats = {
        "reads_seen": 0, "reads_written": 0, "skipped_no_gff": 0, "skipped_ambiguous_base": 0,
        "skipped_quality_check": 0, "skipped_partial_or_pseudo": 0, "contigs_fetched": 0, "taxids_fetched": 0,
    }
    columns = [
        "read_name", "read", "cds_coords", "cds_fragments_connection", "start_coord",
        "contig_accession", "genome_bin_id", "CIGAR", "strand", "uncertain_region_overlap", "domain", "seq_errors",
        "rf0_labels", "rf1_labels", "rf2_labels",
    ]
    fasta_path = out_path.parent / (out_path.name.removesuffix(".tsv.gz") + ".fasta.gz")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", newline="") as out_fh, gzip.open(fasta_path, "wt") as fasta_fh:
        writer = csv.writer(out_fh, delimiter="\t")
        writer.writerow(columns)
        for bam_path in bam_files:
            genome_bin_id = bam_path.stem.split("_", 1)[1] if "_" in bam_path.stem else bam_path.stem
            process_bam(bam_path, genome_bin_id, cds_by_contig, gff_cache_dir, accession_to_taxid, domain_by_taxid,
                        taxonomy_cache_dir, rate_limiter, args.api_key, not args.skip_fetch, writer, fasta_fh, stats, limit=args.limit)
            if args.limit is not None and stats["reads_seen"] >= args.limit:
                break

    print(f"Done. {stats}")
    print(f"TSV written to {out_path}")
    print(f"FASTA written to {fasta_path}")


if __name__ == "__main__":
    main()
