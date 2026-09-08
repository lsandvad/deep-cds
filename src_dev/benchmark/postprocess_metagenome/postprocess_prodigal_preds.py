"""
Postprocess Prodigal predictions on the CAMI metagenome test set (raw_predictions
from predict_metagenome/predict_with_prodigal.py) into the same model_preds_dict.pkl
format postprocess/postprocess_prodigal_preds.py produces for the Mason-simulated
genome testsets - just keyed by CAMI sample instead of genome accession.

process_model_preds itself is unchanged from the original: the "+"-strand-only GFF
convention (predict_with_prodigal.py greps out complement-strand lines the same way
for both pipelines) and the pipe-delimited read_name convention are identical for
CAMI reads. Only the accession/testset-type loop is replaced with dynamic discovery
of whichever CAMI samples were actually predicted, mirroring
processed_predictions/prodigal_preds/CAMI_metagenome/{sample}/ onto
raw_predictions/prodigal_preds/CAMI_metagenome/{sample}/ one-to-one.

EXCLUDE_UNCERTAIN_REGION_READS (below) mirrors postprocess_metagenome_testset.py's
toggle of the same name: when on, any read that postprocess_metagenome_testset.py
would drop from the test set (uncertain_region_overlap=True - overlapping a
hypothetical-protein / ab-initio-prediction-tagged CDS) is also dropped here from the
predictions, so a model isn't penalized as a false positive for calling a CDS on a
read whose ground truth we've decided not to trust.

EXCLUDE_NO_CDS_CONTIG_READS (below) mirrors postprocess_metagenome_testset.py's toggle
of the same name: when on, predictions on reads whose contig has zero CDS annotations
at all (see load_no_cds_contigs / test_CAMI/cami_contigs_without_cds.py) are dropped too,
so a model isn't penalized as a false positive on a contig NCBI simply hasn't annotated.
Independent of and composable with EXCLUDE_UNCERTAIN_REGION_READS.
"""
import argparse
import glob
import os
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# SCARB cluster data root - matches postprocess/postprocess_prodigal_preds.py and every
# other benchmark/postprocess script's project_root convention.
project_root = "/tmp/nrt204/FragmentPredictor"

data_dir = "CAMI_metagenome"  # matches predict_with_prodigal.py / predict_with_fgs.py / predict_with_DeepCDS.py

# See postprocess_metagenome_testset.py's toggles of the same names.
EXCLUDE_UNCERTAIN_REGION_READS = False
EXCLUDE_NO_CDS_CONTIG_READS = False


def load_excluded_reads(sample, project_root=project_root):
    """read_names flagged uncertain_region_overlap=True in the sample's CDS-labeled TSV
    (hypothetical-protein / ab-initio-prediction-tagged ground truth). Only used when
    EXCLUDE_UNCERTAIN_REGION_READS is on. Per-read: a read's mate is unaffected unless
    it is itself tagged."""
    tsv_path = f"{project_root}/data/processed_data/reads_processed/test/{data_dir}/{sample}_cds_labels.tsv.gz"
    df = pd.read_csv(tsv_path, sep="\t", compression="gzip", usecols=["read_name", "uncertain_region_overlap"])
    return set(df.loc[df["uncertain_region_overlap"], "read_name"])


def load_no_cds_contigs(sample, project_root=project_root):
    """Contig accessions with zero CDS features in their raw GFF3 at all, persisted by
    test_CAMI/cami_contigs_without_cds.py."""
    contigs_path = Path(project_root) / "data" / "processed_data" / "CAMI_metagenome_no_cds_contigs" / f"{sample}_contigs_without_cds.txt"
    if not contigs_path.exists():
        raise FileNotFoundError(
            f"{contigs_path} not found - run test_CAMI/cami_contigs_without_cds.py --sample {sample} first."
        )
    return set(contigs_path.read_text().split())


def load_no_cds_contig_reads(sample, project_root=project_root):
    """read_names on a contig with zero CDS annotations at all (see load_no_cds_contigs).
    Only used when EXCLUDE_NO_CDS_CONTIG_READS is on."""
    no_cds_contigs = load_no_cds_contigs(sample, project_root=project_root)
    tsv_path = f"{project_root}/data/processed_data/reads_processed/test/{data_dir}/{sample}_cds_labels.tsv.gz"
    df = pd.read_csv(tsv_path, sep="\t", compression="gzip", usecols=["read_name", "contig_accession"])
    return set(df.loc[df["contig_accession"].isin(no_cds_contigs), "read_name"])


def process_model_preds(sample, excluded_reads=frozenset()):
    """
    Process Prodigal predictions from a GFF file for a given CAMI sample.

    Args:
        sample (str): The CAMI sample identifier, e.g. "sample_0".
        excluded_reads: read_names to drop from the result entirely (see
            EXCLUDE_UNCERTAIN_REGION_READS / load_excluded_reads).

    Returns:
        model_dict (dict): A dictionary where keys are read names and values are dictionaries with 'cds_coords' (CDS coordinates).
    """
    # Initialize
    model_dict = dict()

    # Read model predictions GFF file
    with open(f"{project_root}/data/processed_data/predictions/raw_predictions/prodigal_preds/{data_dir}/{sample}/{sample}.gff", "r") as file:
        file.readline()  # Skip first line

        # Get CDS predictions for read
        for line in file:
            read_name = line.split("\t")[0].split("|")[0]
            attr_type = line.split("\t")[2]
            strand = line.split("\t")[6]
            assert strand == "+", "Complement strand predictions not filtered out properly!"

            cds_coords = []

            if attr_type == "CDS":

                if read_name not in model_dict.keys():
                    model_dict[read_name] = dict()
                    model_dict[read_name]["cds_coords"] = []
                    model_dict[read_name]["cds_fragments_connection"] = []

                    counter_cds_on_read = 0

                # Get CDS coordinates and reading frame
                cds_start = int(line.split("\t")[3])
                cds_end = int(line.split("\t")[4])

                if cds_start % 3 == 1:
                    rf = 0
                elif cds_start % 3 == 2:
                    rf = 1
                elif cds_start % 3 == 0:
                    rf = 2

                cds_coords = [cds_start, cds_end, str(rf)]

                # Save sequences of length 60 or more (prodigal can only go down to this length,
                # but due to FGS' processing (does not predict the last codon in RF0),
                # we miss some sequences which are also removed from test set)
                if cds_end - cds_start + 1 > 60:
                    model_dict[read_name]["cds_coords"].append(cds_coords)
                    model_dict[read_name]["cds_fragments_connection"].append([counter_cds_on_read])
                    counter_cds_on_read += 1

    # Reorganize to only include reads with coding sequences
    model_dict = {
        read_name: data
        for read_name, data in model_dict.items()
        if data["cds_coords"] and read_name not in excluded_reads
    }

    return model_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-uncertain-region-reads", action="store_true", default=EXCLUDE_UNCERTAIN_REGION_READS,
                         help="Override EXCLUDE_UNCERTAIN_REGION_READS (module default: %(default)s)")
    parser.add_argument("--exclude-no-cds-contig-reads", action="store_true", default=EXCLUDE_NO_CDS_CONTIG_READS,
                         help="Override EXCLUDE_NO_CDS_CONTIG_READS (module default: %(default)s)")
    args = parser.parse_args()
    EXCLUDE_UNCERTAIN_REGION_READS = args.exclude_uncertain_region_reads
    EXCLUDE_NO_CDS_CONTIG_READS = args.exclude_no_cds_contig_reads

    input_dir = f"{project_root}/data/processed_data/reads_processed/test/{data_dir}"
    samples = sorted(
        os.path.basename(p).replace("_cds_labels.tsv.gz", "")
        for p in glob.glob(f"{input_dir}/*_cds_labels.tsv.gz")
    )
    if not samples:
        raise FileNotFoundError(f"No *_cds_labels.tsv.gz files found in: {input_dir}")

    # Output-only directory name - raw predictions are unaffected by the toggles, only
    # which reads make it into processed_predictions/ here. Keeps all versions from
    # overwriting each other so they remain available for comparison.
    out_data_dir = (
        data_dir
        + ("_no_ab_initio" if EXCLUDE_UNCERTAIN_REGION_READS else "")
        + ("_no_cds_contigs" if EXCLUDE_NO_CDS_CONTIG_READS else "")
    )
    print(f"EXCLUDE_UNCERTAIN_REGION_READS={EXCLUDE_UNCERTAIN_REGION_READS}, "
          f"EXCLUDE_NO_CDS_CONTIG_READS={EXCLUDE_NO_CDS_CONTIG_READS} -> writing to processed_predictions/prodigal_preds/{out_data_dir}/")

    for sample in tqdm(samples, desc="Processing Prodigal predictions for CAMI samples..."):
        print(sample)

        out_dir = f"{project_root}/data/processed_data/predictions/processed_predictions/prodigal_preds/{out_data_dir}/{sample}"
        os.makedirs(out_dir, exist_ok=True)

        excluded_reads = set()
        if EXCLUDE_UNCERTAIN_REGION_READS:
            excluded_reads |= load_excluded_reads(sample)
        if EXCLUDE_NO_CDS_CONTIG_READS:
            excluded_reads |= load_no_cds_contig_reads(sample)
        model_preds_dict = process_model_preds(sample, excluded_reads=excluded_reads)

        with open(f"{out_dir}/model_preds_dict.pkl", "wb") as processed_preds_file:
            pickle.dump(model_preds_dict, processed_preds_file)
