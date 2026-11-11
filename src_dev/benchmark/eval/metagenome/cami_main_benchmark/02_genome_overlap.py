"""Step 2: overlap between CAMI III genomes and the DeepCDS training/validation genomes.

For every genome bin in the CAMI sample:
  1. its source genome file (CAMI source_genomes/*.fna), found via the contig accessions of its reads
     (labels TSV: genome_bin_id, contig_accession) and the FASTA headers;
  2. same assembly: does any of its contig accessions occur in a train/val assembly
     (sequence_report.jsonl: refseqAccession / genbankAccession)?
  3. for Bacteria/Archaea: ANI to every train/val genome with skani (skani dist, default settings;
     pairs below ~80% ANI are not reported). The best hit = highest ANI.
Eukaryota/Viruses/Unknown bins cannot overlap with the prokaryotic training set; ANI is not computed.

Categories (per bin): identical = same assembly or ANI >= 99.9%; same species = ANI >= 95%;
novel = otherwise (incl. no skani hit).

Outputs: $RESULTS/cami_genome_overlap.tsv; skani raw output in $WORK/skani/
Usage: python 02_genome_overlap.py [--work /tmp/nrt204/cami_overlap_work] [--threads 128]
"""
import argparse
import glob
import json
import os
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = "/tmp/nrt204/FragmentPredictor/data/processed_data/cami_main_benchmark"
project_root = "/tmp/nrt204/FragmentPredictor"
SKANI = "/home/nrt204/.conda/envs/skani_env/bin/skani"
IDENTICAL_ANI, SPECIES_ANI = 99.9, 95.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/cami_overlap_work")
    parser.add_argument("--threads", type=int, default=128)
    args = parser.parse_args()
    w = args.work

    # 1. bins -> contigs (all reads of the sample)
    meta = pd.read_csv(f"{project_root}/data/processed_data/reads_processed/test/CAMI_metagenome/sample_0_cds_labels.tsv.gz",
                       sep="\t", compression="gzip", usecols=["genome_bin_id", "contig_accession", "domain"])
    bin_contigs = meta.drop_duplicates()
    del meta
    print(f"{bin_contigs['genome_bin_id'].nunique()} genome bins, {bin_contigs['contig_accession'].nunique()} contigs")

    # contigs -> CAMI source genome file
    contig_file = {}
    for path in glob.glob(f"{w}/cami_genomes/**/*.fna", recursive=True):
        with open(path) as f:
            for line in f:
                if line.startswith(">"):
                    contig_file[line[1:].split()[0]] = path
    bin_contigs["source_file"] = bin_contigs["contig_accession"].map(contig_file)
    bins = (bin_contigs.groupby("genome_bin_id")
            .agg(domain=("domain", "first"), n_contigs=("contig_accession", "nunique"),
                 source_file=("source_file", lambda s: s.mode().iloc[0] if s.notna().any() else None),
                 n_source_files=("source_file", "nunique"))
            .reset_index())
    print(f"bins without a source genome file: {bins['source_file'].isna().sum()}; "
          f"bins whose contigs span > 1 file: {(bins['n_source_files'] > 1).sum()}")

    # 2. same assembly: CAMI contig accession present in a train/val assembly
    train_contig_to_acc = {}
    for path in glob.glob(f"{w}/train_val_genomes/*.sequence_report.jsonl"):
        acc = os.path.basename(path).split(".sequence_report")[0]
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                for key in ("refseqAccession", "genbankAccession"):
                    if rec.get(key):
                        train_contig_to_acc[rec[key]] = acc
    bin_contigs["train_assembly"] = bin_contigs["contig_accession"].map(train_contig_to_acc)
    same = bin_contigs.dropna(subset=["train_assembly"]).groupby("genome_bin_id")["train_assembly"].first()
    bins["same_assembly_as"] = bins["genome_bin_id"].map(same)
    print(f"bins sharing contigs with a train/val assembly: {bins['same_assembly_as'].notna().sum()}")

    # 3. skani: prokaryotic CAMI genomes vs train/val genomes
    os.makedirs(f"{w}/skani", exist_ok=True)
    prok = bins[bins["domain"].isin(["Bacteria", "Archaea"]) & bins["source_file"].notna()]
    queries = sorted(prok["source_file"].unique())
    refs = sorted(glob.glob(f"{w}/train_val_genomes/*.fna"))
    with open(f"{w}/skani/queries.txt", "w") as f:
        f.write("\n".join(queries) + "\n")
    with open(f"{w}/skani/refs.txt", "w") as f:
        f.write("\n".join(refs) + "\n")
    out = f"{w}/skani/skani_dist.tsv"
    if not os.path.exists(out):
        print(f"skani: {len(queries)} CAMI genomes x {len(refs)} train/val genomes")
        subprocess.run([SKANI, "dist", "--ql", f"{w}/skani/queries.txt", "--rl", f"{w}/skani/refs.txt",
                        "-t", str(args.threads), "-o", out], check=True)
    sk = pd.read_csv(out, sep="\t")
    sk["train_genome"] = sk["Ref_file"].map(lambda p: os.path.basename(p)[:-4])
    best = (sk.sort_values("ANI", ascending=False).drop_duplicates("Query_file")
            .set_index("Query_file")[["ANI", "Align_fraction_query", "Align_fraction_ref", "train_genome"]])
    bins = bins.join(best, on="source_file")

    # Taxonomy of the best training hit
    info = pd.read_csv(f"{project_root}/data/processed_data/dataset_information/genomes_info_with_partitions.csv").set_index("accession")
    for col in ("species", "family", "partition"):
        bins[f"train_{col}"] = bins["train_genome"].map(info[col]) if col in info.columns else None

    bins["category"] = "novel"
    bins.loc[bins["ANI"] >= SPECIES_ANI, "category"] = "same species"
    bins.loc[(bins["ANI"] >= IDENTICAL_ANI) | bins["same_assembly_as"].notna(), "category"] = "identical"
    bins.loc[~bins["domain"].isin(["Bacteria", "Archaea"]), "category"] = "non-prokaryotic"

    os.makedirs(f"{RESULTS}", exist_ok=True)
    bins.to_csv(f"{RESULTS}/cami_genome_overlap.tsv", sep="\t", index=False)
    print(bins["category"].value_counts().to_string())
    print("STEP 2 DONE")


if __name__ == "__main__":
    main()
