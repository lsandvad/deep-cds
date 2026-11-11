"""Step 6: CAMI III sample_0 dataset statistics, with and without the genomes that overlap
the model-development (train/val) genomes (category identical or same species in $RESULTS/cami_genome_overlap.tsv).

Original: per-genome read counts from the sample_0 BAMs (raw_data/CAMI_datasets/sample_0_bam/, one BAM per genome,
primary records = reads; read pairs = reads / 2). Genome domains for the original data are not available for the
9 plasmids missing from the cleaned set, so - as in the existing table - no domain split is given there.
Cleaned (Base / Excl. uncertain annotations / Excl. contigs lacking annotations): the read list of each test-set
variant (testset_processed/<variant>/sample_0/read_names_list.pkl) joined with the per-read metadata
(reads_processed/test/CAMI_metagenome/sample_0_cds_labels.tsv.gz): read pairs = distinct <contig>-<pair id>
(read name before "_seq"), reads with a labeled CDS = cds_coords not empty, genomes = distinct genome_bin_id.
Check: the "all genomes" numbers must equal the published table.
Output: $RESULTS/dataset_statistics.tsv (printed as LaTeX rows)
Usage: python 06_dataset_statistics.py
"""
import os
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = "/tmp/nrt204/FragmentPredictor/data/processed_data/cami_main_benchmark"
D = "/tmp/nrt204/FragmentPredictor/data"
BAM_DIR = f"{D}/raw_data/CAMI_datasets/sample_0_bam"
SAMTOOLS = os.path.expanduser("~/.conda/envs/poc_env/bin/samtools")
VARIANTS = {"Base": "CAMI_metagenome", "Excl. uncertain annotations": "CAMI_metagenome_no_ab_initio",
            "Excl. contigs lacking annotations": "CAMI_metagenome_no_cds_contigs"}
DOMAINS = ["Bacteria", "Eukaryota", "Viruses"]
# previously published statistics (all genomes), for the check
PUBLISHED = {
    "Original": {"read_pairs": 16_633_513, "reads": 33_267_026, "genomes": 309},
    "Base": {"read_pairs": 16_518_313, "reads": 32_680_743, "reads_Bacteria": 27_363_884, "reads_Eukaryota": 4_795_966,
             "reads_Viruses": 520_893, "reads_labeled": 13_203_463, "genomes": 300, "genomes_Bacteria": 156,
             "genomes_Eukaryota": 11, "genomes_Viruses": 133},
    "Excl. uncertain annotations": {"read_pairs": 15_187_972, "reads": 31_343_322, "reads_Bacteria": 26_146_341,
                                    "reads_Eukaryota": 4_795_966, "reads_Viruses": 401_015, "reads_labeled": 11_869_796,
                                    "genomes": 300, "genomes_Bacteria": 156, "genomes_Eukaryota": 11, "genomes_Viruses": 133},
    "Excl. contigs lacking annotations": {"read_pairs": 14_357_367, "reads": 28_358_851, "reads_Bacteria": 27_336_766,
                                          "reads_Eukaryota": 501_192, "reads_Viruses": 520_893, "reads_labeled": 13_203_463,
                                          "genomes": 290, "genomes_Bacteria": 156, "genomes_Eukaryota": 1,
                                          "genomes_Viruses": 133},
}


def bam_reads(path):
    out = subprocess.run([SAMTOOLS, "view", "-c", "-F", "0x900", path], capture_output=True, text=True, check=True)
    return os.path.basename(path)[len("sample0_"):-len(".bam")], int(out.stdout)


def stats(meta):
    s = {"read_pairs": meta["read_name"].str.rsplit("_seq", n=1).str[0].nunique(), "reads": len(meta)}
    for d in DOMAINS:
        s[f"reads_{d}"] = int((meta["domain"] == d).sum())
    s["reads_labeled"] = int((meta["cds_coords"] != "[]").sum())
    g = meta.drop_duplicates("genome_bin_id")
    s["genomes"] = len(g)
    for d in DOMAINS:
        s[f"genomes_{d}"] = int((g["domain"] == d).sum())
    return s


def main():
    overlap = pd.read_csv(f"{RESULTS}/cami_genome_overlap.tsv", sep="\t")
    removed = set(overlap.loc[overlap["category"].isin(["identical", "same species"]), "genome_bin_id"])
    print(f"genomes overlapping train/val (identical + same species): {len(removed)}")

    with ThreadPoolExecutor(32) as ex:
        bam = dict(ex.map(bam_reads, sorted(f"{BAM_DIR}/{f}" for f in os.listdir(BAM_DIR) if f.endswith(".bam"))))
    bam = {g: n for g, n in bam.items() if n > 0}
    cols = {}
    for label, keep in [("Original", bam), ("Original, overlap removed", {g: n for g, n in bam.items() if g not in removed})]:
        cols[label] = {"read_pairs": sum(keep.values()) // 2, "reads": sum(keep.values()), "genomes": len(keep)}

    meta = pd.read_csv(f"{D}/processed_data/reads_processed/test/CAMI_metagenome/sample_0_cds_labels.tsv.gz", sep="\t",
                       usecols=["read_name", "cds_coords", "genome_bin_id", "domain"])
    for label, variant in VARIANTS.items():
        with open(f"{D}/processed_data/testset_processed/{variant}/sample_0/read_names_list.pkl", "rb") as f:
            names = set(pickle.load(f))
        sub = meta[meta["read_name"].isin(names)]
        assert len(sub) == len(names), (variant, len(sub), len(names))
        cols[label] = stats(sub)
        cols[f"{label}, overlap removed"] = stats(sub[~sub["genome_bin_id"].isin(removed)])

    mismatches = {label: {k: (cols[label][k], v) for k, v in published.items() if cols[label][k] != v}
                  for label, published in PUBLISHED.items()}
    mismatches = {k: v for k, v in mismatches.items() if v}
    print("Check against the published table (computed, published):", mismatches or "all values reproduced")
    for label in cols:
        if "uncertain" in label or "lacking" in label:
            sub = meta[meta["read_name"].isin(set(pickle.load(open(
                f"{D}/processed_data/testset_processed/{VARIANTS[label.split(',')[0]]}/sample_0/read_names_list.pkl", "rb"))))]
            if "overlap removed" in label:
                sub = sub[~sub["genome_bin_id"].isin(removed)]
            pair = sub["read_name"].str.rsplit("_seq", n=1).str[0]
            cols[label]["read_pairs_both_mates"] = int((pair.value_counts() == 2).sum())

    table = pd.DataFrame(cols)
    table.to_csv(f"{RESULTS}/dataset_statistics.tsv", sep="\t")
    print(table.to_string())


if __name__ == "__main__":
    main()
