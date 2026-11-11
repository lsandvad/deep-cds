"""
Community-composition statistics for a CAMI sample's simulated reads - the kind of
numbers a Methods section needs (how many genomes/taxa the reads were drawn from, how
they're distributed across domains, how many survived CDS labeling), computed straight
from the same raw/processed files the rest of the pipeline uses, not re-derived by hand.

Two sources, on purpose:
  - reads_mapping.tsv.gz (raw CAMI gold-standard read->genome mapping): the authoritative
    ground truth for "how many genomes/taxa contributed reads to this sample" - computed
    BEFORE any of our own alignment/labeling filtering, so it reflects the simulated
    community as CAMI built it.
  - <sample>_cds_labels.tsv.gz (this repo's own labeling output, from
    process_and_map_CAMI_testset.py): adds the domain breakdown (Bacteria/Archaea/...)
    and shows how many reads/genomes actually survived labeling QC, which will always be
    somewhat less than the raw mapping - useful to report alongside the raw count, not
    instead of it.

Usage:
    python3 cami_sample_composition_stats.py --sample sample_0
"""
import argparse
import glob
import gzip
import os

import numpy as np
import pandas as pd

# SCARB cluster data root - matches process_and_map_CAMI_testset.py's --data-root default
# and postprocess_metagenome_testset.py's project_root convention.
SCARB_ROOT = "/tmp/nrt204/FragmentPredictor"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--sample", default="sample_0", help="CAMI sample name, e.g. sample_0 (default: sample_0)")
parser.add_argument("--data-root", default=f"{SCARB_ROOT}/data/raw_data/CAMI_datasets",
                     help="Raw CAMI data directory (default: matches process_and_map_CAMI_testset.py)")
parser.add_argument("--project-root", default=SCARB_ROOT,
                     help="Project root for locating the processed/labeled TSV (default: SCARB root)")
parser.add_argument("--chunksize", type=int, default=2_000_000,
                     help="Rows per chunk when streaming the large mapping/labeled TSVs (default: 2000000)")


def count_reference_genomes(data_root):
    """Total genomes in the CAMI reference community pool (source_genomes/*.fna),
    for context - only a subset of these typically end up with reads in any one sample."""
    genome_files = glob.glob(os.path.join(data_root, "source_genomes", "*.fna"))
    return len(genome_files)


def count_bam_files(data_root, sample):
    """Per-genome BAM file count for this sample - should match reads_mapping's distinct
    genome_id count; reported separately so a mismatch (e.g. a genome with a BAM file but
    zero surviving mapped reads) is visible rather than silently picking one number."""
    bam_files = glob.glob(os.path.join(data_root, f"{sample}_bam", "*.bam"))
    return len(bam_files)


def summarize_reads_mapping(mapping_path, chunksize):
    """Stream reads_mapping.tsv.gz (columns: anonymous_read_id, genome_id, tax_id,
    read_id) and accumulate the stats that require a full pass, without loading the
    whole (tens-of-millions-of-rows) file into memory at once.

    Rows here are individual mates (anonymous_read_id like "S0R0/1", "S0R0/2" - same
    template, two rows), so total_read_records is an individual-read count. Read PAIRS
    are counted separately, by stripping the trailing "/1" or "/2" and counting distinct
    template IDs - this is the raw simulator output, before any of our own filtering, so
    every template is expected to have exactly both mates present (no orphaning yet);
    that's cross-checked below rather than assumed.

    Returns:
        dict with total_read_records, total_read_pairs, distinct_genome_ids,
        distinct_tax_ids, and reads_per_genome (a pandas Series of
        genome_id -> individual-read-record count).
    """
    # The file's actual header is "#anonymous_read_id" (literal leading '#', not a
    # comment marker - pandas needs the exact name in usecols to find the column).
    anon_read_id_col = "#anonymous_read_id"

    total_read_records = 0
    reads_per_genome = pd.Series(dtype="int64")
    tax_ids = set()
    pair_ids = set()

    for chunk in pd.read_csv(mapping_path, sep="\t", usecols=[anon_read_id_col, "genome_id", "tax_id"],
                              chunksize=chunksize):
        total_read_records += len(chunk)
        reads_per_genome = reads_per_genome.add(chunk["genome_id"].value_counts(), fill_value=0)
        tax_ids.update(chunk["tax_id"].unique().tolist())
        # anonymous_read_id is always "<template>/1" or "<template>/2" - strip the
        # 2-char mate suffix to recover the shared template ID.
        pair_ids.update(chunk[anon_read_id_col].str[:-2].unique().tolist())

    reads_per_genome = reads_per_genome.astype("int64")

    return {
        "total_read_records": total_read_records,
        "total_read_pairs": len(pair_ids),
        "distinct_genome_ids": reads_per_genome.shape[0],
        "distinct_tax_ids": len(tax_ids),
        "reads_per_genome": reads_per_genome,
    }


def load_no_cds_contigs(sample, project_root):
    """Contig accessions with zero CDS features in their raw GFF3 at all, persisted by
    src_dev/benchmark/eval/metagenome/cami_sample_statistics/cami_contigs_without_cds.py. Returns None (not an empty set) if that script
    hasn't been run yet for this sample, so callers can tell "not computed" apart from
    "computed, none found"."""
    contigs_path = os.path.join(
        project_root, "data", "processed_data", "CAMI_metagenome_no_cds_contigs", f"{sample}_contigs_without_cds.txt"
    )
    if not os.path.isfile(contigs_path):
        return None
    with open(contigs_path) as f:
        return set(f.read().split())


def summarize_processed_labels(labels_path, chunksize, no_cds_contigs=None):
    """Stream <sample>_cds_labels.tsv.gz (this repo's labeling output) and summarize how
    many reads/genomes survived labeling QC, broken down by domain - computed for "all"
    (every labeled read), "excluding" (what postprocess_metagenome_testset.py's
    EXCLUDE_UNCERTAIN_REGION_READS=True keeps), and - if no_cds_contigs is given (see
    load_no_cds_contigs) - "excluding_no_cds_contigs" (what its EXCLUDE_NO_CDS_CONTIG_READS=True
    keeps: every read on a contig with zero CDS annotations at all is dropped).

    Two different conventions are used for individual reads vs. read pairs, deliberately,
    for BOTH exclusion criteria:
      - Individual reads / coding reads / genomes / domain breakdown: per-read. A read is
        excluded only if it ITSELF is flagged uncertain_region_overlap=True (or, for the
        contig criterion, its own contig_accession is in no_cds_contigs); its mate is
        retained (orphaned) if it isn't itself excluded. This matches what each toggle
        actually keeps in the filtered data.
      - Read pairs: per-pair, stricter. A pair counts as "removed" if EITHER mate is
        excluded, even though the clean mate is still physically retained under the
        per-read rule above - this reports how many original sequenced fragments were
        touched at all, a different question from what's literally kept in the filtered
        dataset. For the contig criterion this isn't a no-op: a fragment can straddle a
        contig boundary in a draft assembly, so a pair's two mates aren't guaranteed to
        share a contig_accession. This doesn't depend on the per-read policy above, so
        it's computed once and used regardless of it.

    Rows are individual mates (read_name like "<query_name>_seq1", "<query_name>_seq2" -
    see process_and_map_CAMI_testset.py's seq_id construction); read PAIRS are identified
    by stripping the trailing "_seq<N>" counter to recover query_name.

    Returns:
        dict with keys "all", "excluding", and (if no_cds_contigs given)
        "excluding_no_cds_contigs" (each a dict with total_labeled_reads, coding_reads -
        reads with a non-empty cds_coords, i.e. an actual labeled CDS, not just a read
        that survived QC - distinct_genomes_labeled, and a per-domain DataFrame of
        (n_reads, n_genomes)), plus "pair_impact" (ab-initio: total_pairs, pairs_clean,
        pairs_touched) and, if no_cds_contigs given, "pair_impact_no_cds_contigs" (same
        shape, for the contig criterion).
    """
    usecols = ["read_name", "genome_bin_id", "domain", "uncertain_region_overlap", "cds_coords"]
    if no_cds_contigs is not None:
        usecols = usecols + ["contig_accession"]

    variants = ["all", "excluding"]
    if no_cds_contigs is not None:
        variants.append("excluding_no_cds_contigs")
    acc = {
        v: {"total_reads": 0, "coding_reads": 0, "genomes_by_domain": {}, "reads_by_domain": {}}
        for v in variants
    }
    all_pair_ids = set()
    tagged_pair_ids = set()
    no_cds_pair_ids = set()

    for chunk in pd.read_csv(labels_path, sep="\t", usecols=usecols, chunksize=chunksize):
        is_coding = chunk["cds_coords"] != "[]"
        pair_id = chunk["read_name"].str.rsplit("_seq", n=1).str[0]
        tagged_mask = chunk["uncertain_region_overlap"]
        masks = {"all": pd.Series(True, index=chunk.index), "excluding": ~tagged_mask}

        if no_cds_contigs is not None:
            no_cds_mask = chunk["contig_accession"].isin(no_cds_contigs)
            masks["excluding_no_cds_contigs"] = ~no_cds_mask

        for variant, mask in masks.items():
            sub = chunk[mask]
            a = acc[variant]
            a["total_reads"] += len(sub)
            a["coding_reads"] += int(is_coding[mask].sum())
            for domain, group in sub.groupby("domain"):
                a["genomes_by_domain"].setdefault(domain, set()).update(group["genome_bin_id"].unique().tolist())
                a["reads_by_domain"][domain] = a["reads_by_domain"].get(domain, 0) + len(group)

        all_pair_ids.update(pair_id.unique().tolist())
        if tagged_mask.any():
            tagged_pair_ids.update(pair_id[tagged_mask].unique().tolist())
        if no_cds_contigs is not None and no_cds_mask.any():
            no_cds_pair_ids.update(pair_id[no_cds_mask].unique().tolist())

    results = {}
    for variant, a in acc.items():
        all_genomes = set().union(*a["genomes_by_domain"].values()) if a["genomes_by_domain"] else set()
        domain_table = pd.DataFrame({
            "n_reads": pd.Series(a["reads_by_domain"]),
            "n_genomes": pd.Series({d: len(g) for d, g in a["genomes_by_domain"].items()}),
        }).sort_values("n_reads", ascending=False)
        results[variant] = {
            "total_labeled_reads": a["total_reads"],
            "coding_reads": a["coding_reads"],
            "distinct_genomes_labeled": len(all_genomes),
            "domain_table": domain_table,
        }

    results["pair_impact"] = {
        "total_pairs": len(all_pair_ids),
        "pairs_touched": len(tagged_pair_ids),
        "pairs_clean": len(all_pair_ids - tagged_pair_ids),
    }
    if no_cds_contigs is not None:
        results["pair_impact_no_cds_contigs"] = {
            "total_pairs": len(all_pair_ids),
            "pairs_touched": len(no_cds_pair_ids),
            "pairs_clean": len(all_pair_ids - no_cds_pair_ids),
        }
    return results


def main():
    args = parser.parse_args()

    mapping_path = os.path.join(args.data_root, f"{args.sample}_reads", "reads_mapping.tsv.gz")
    labels_path = os.path.join(
        args.project_root, "data", "processed_data", "reads_processed", "test",
        "CAMI_metagenome", f"{args.sample}_cds_labels.tsv.gz",
    )

    print(f"=== CAMI sample composition: {args.sample} ===\n")

    n_reference_genomes = count_reference_genomes(args.data_root)
    n_bam_files = count_bam_files(args.data_root, args.sample)
    print(f"Reference community pool:        {n_reference_genomes:,} genomes ({args.data_root}/source_genomes/)")
    print(f"BAM files for this sample:        {n_bam_files:,} ({args.data_root}/{args.sample}_bam/)")

    if not os.path.isfile(mapping_path):
        print(f"\n[skip] reads_mapping.tsv.gz not found at {mapping_path}")
    else:
        print(f"\nReading gold-standard mapping ({mapping_path}) - this scans the full file, may take a minute...")
        mapping_stats = summarize_reads_mapping(mapping_path, args.chunksize)
        reads_per_genome = mapping_stats["reads_per_genome"]

        print(f"\nDistinct genomes with simulated reads:  {mapping_stats['distinct_genome_ids']:,}"
              f" (of {n_reference_genomes:,} in the reference pool)")
        if mapping_stats["distinct_genome_ids"] != n_bam_files:
            print(f"  [note] BAM file count ({n_bam_files:,}) differs from distinct genome_id count "
                  f"in the mapping ({mapping_stats['distinct_genome_ids']:,}) - worth checking which "
                  f"genome(s) account for the difference before citing either number.")
        print(f"Distinct taxonomic IDs (tax_id):        {mapping_stats['distinct_tax_ids']:,}")
        print(f"\nBEFORE (raw simulated output, gold-standard mapping):")
        print(f"  Individual reads (mates):  {mapping_stats['total_read_records']:,}")
        print(f"  Read pairs (fragments):    {mapping_stats['total_read_pairs']:,}")
        if mapping_stats["total_read_records"] != 2 * mapping_stats["total_read_pairs"]:
            print(f"  [note] Not every template has exactly 2 mates in the raw mapping - "
                  f"{mapping_stats['total_read_records'] - 2 * mapping_stats['total_read_pairs']:+,} "
                  f"vs. a perfectly paired file.")

        print(f"\nReads per genome (raw, individual-read/mate records):")
        print(f"  min    = {int(reads_per_genome.min()):,}")
        print(f"  median = {int(reads_per_genome.median()):,}")
        print(f"  mean   = {reads_per_genome.mean():,.1f}")
        print(f"  max    = {int(reads_per_genome.max()):,}")

    if not os.path.isfile(labels_path):
        print(f"\n[skip] Labeled TSV not found at {labels_path}")
        print("  (run process_and_map_CAMI_testset.py first for the domain breakdown / QC-survival numbers)")
    else:
        no_cds_contigs = load_no_cds_contigs(args.sample, args.project_root)
        if no_cds_contigs is None:
            print(f"\n[note] No no-CDS-contigs list found for {args.sample} - run "
                  f"cami_contigs_without_cds.py --sample {args.sample} first to also report the "
                  f"'excluding annotation-free contigs' variant below.")
        else:
            print(f"\n{len(no_cds_contigs):,} contigs with zero CDS annotations at all "
                  f"(from cami_contigs_without_cds.py).")

        print(f"\nReading labeled TSV ({labels_path}) - this also scans the full file...")
        label_results = summarize_processed_labels(labels_path, args.chunksize, no_cds_contigs=no_cds_contigs)
        including = label_results["all"]
        excluding = label_results["excluding"]

        pi = label_results["pair_impact"]

        print(f"\nDistinct genomes represented after QC:  {including['distinct_genomes_labeled']:,}")
        print(f"\nAFTER (survived alignment + labeling QC):")
        print(f"  Individual reads (mates):  {including['total_labeled_reads']:,}")
        print(f"  Read pairs (fragments):    {pi['total_pairs']:,}")

        print(f"\nBy domain (individual reads):")
        print(including["domain_table"].to_string())

        print(f"\n=== Including vs excluding ab-initio-tagged CDS regions ===")
        print(f"(CDS_TAG_MARKERS: 'product=hypothetical protein' / 'ab initio prediction' - see")
        print(f" process_and_map_CAMI_testset.py. 'Excluding' = what")
        print(f" postprocess_metagenome_testset.py's EXCLUDE_UNCERTAIN_REGION_READS=True keeps:")
        print(f" a read is dropped if IT is tagged; its mate is kept (orphaned) if it isn't")
        print(f" itself tagged. 'Read pairs' below uses a stricter, separate convention:")
        print(f" a pair counts as removed if EITHER mate is tagged, even though the clean")
        print(f" mate is still retained above - this reports how many original sequenced")
        print(f" fragments were touched at all, not what's literally kept in the data.)\n")

        def comparison_row(label, inc, exc):
            removed = inc - exc
            pct = (removed / inc * 100) if inc else 0.0
            print(f"  {label:<30} incl={inc:>13,}   excl={exc:>13,}   removed={removed:>11,} ({pct:.2f}%)")

        comparison_row("Individual reads (mates)", including["total_labeled_reads"], excluding["total_labeled_reads"])
        comparison_row("Read pairs (fragments)", pi["total_pairs"], pi["pairs_clean"])
        comparison_row("Reads with a labeled CDS", including["coding_reads"], excluding["coding_reads"])
        comparison_row("Distinct genomes represented", including["distinct_genomes_labeled"], excluding["distinct_genomes_labeled"])

        print(f"\nBy domain, EXCLUDING ab-initio-tagged (individual reads, per-read convention):")
        print(excluding["domain_table"].to_string())

        if no_cds_contigs is not None:
            excl_contigs = label_results["excluding_no_cds_contigs"]
            pi_contigs = label_results["pair_impact_no_cds_contigs"]

            print(f"\n=== Including vs excluding reads on annotation-free contigs ===")
            print(f"(Contigs with ZERO CDS features in their raw GFF3 at all - see")
            print(f" src_dev/benchmark/eval/metagenome/cami_sample_statistics/cami_contigs_without_cds.py. 'Excluding' = what")
            print(f" postprocess_metagenome_testset.py's EXCLUDE_NO_CDS_CONTIG_READS=True keeps:")
            print(f" every read on such a contig is dropped, per-read (a read's mate is kept,")
            print(f" orphaned, if it isn't itself on a no-CDS contig). 'Read pairs' below uses")
            print(f" the same stricter, separate convention as above: a pair counts as removed")
            print(f" if EITHER mate is on such a contig - not necessarily both, since a fragment")
            print(f" can straddle a contig boundary in a draft assembly.)\n")

            comparison_row("Individual reads (mates)", including["total_labeled_reads"], excl_contigs["total_labeled_reads"])
            comparison_row("Read pairs (fragments)", pi_contigs["total_pairs"], pi_contigs["pairs_clean"])
            comparison_row("Reads with a labeled CDS", including["coding_reads"], excl_contigs["coding_reads"])
            comparison_row("Distinct genomes represented", including["distinct_genomes_labeled"], excl_contigs["distinct_genomes_labeled"])
            if including["coding_reads"] != excl_contigs["coding_reads"]:
                print(f"  [note] Coding-read count changed - a contig on the no-CDS list must have "
                      f"contributed >=1 labeled CDS, which shouldn't be possible by construction; worth "
                      f"a look before citing either number.")

            print(f"\nBy domain, EXCLUDING annotation-free contigs (individual reads, per-read convention):")
            print(excl_contigs["domain_table"].to_string())

    print()


if __name__ == "__main__":
    main()

