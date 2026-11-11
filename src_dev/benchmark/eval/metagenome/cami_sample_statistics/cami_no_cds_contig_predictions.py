"""
For contigs with zero CDS annotations at all (see cami_contigs_without_cds.py), how much
does each prediction method (Prodigal, FragGeneScanRs, DeepCDS) still call as coding on
their reads?

Distinct from the EXCLUDE_NO_CDS_CONTIG_READS toggle in postprocess_metagenome_testset.py
and postprocess_metagenome/postprocess_*_preds.py (which REMOVES these reads from evaluation
entirely, for both the test set and predictions): this script does the opposite - it isolates
ONLY these reads and reports what each method predicted on them, as a positive-control /
false-discovery check on contigs NCBI hasn't annotated. A high hit rate here could mean either
genuine (unannotated) coding potential a method correctly recovers, or overprediction - this
script can't distinguish the two, only quantify how often it happens per method.

Reads baseline (both EXCLUDE_UNCERTAIN_REGION_READS and EXCLUDE_NO_CDS_CONTIG_READS off)
processed_predictions/ output for each method - run the postprocess_metagenome/postprocess_*.py
scripts with their defaults first if these aren't populated yet. Also requires
cami_contigs_without_cds.py to have already been run for the sample (reads back its persisted
<sample>_contigs_without_cds.txt).

Usage:
    python3 cami_no_cds_contig_predictions.py --sample sample_0
"""
import argparse
import glob
from pathlib import Path

import pandas as pd
import pickle

SCARB_ROOT = Path("/tmp/nrt204/FragmentPredictor")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--sample", default="sample_0", help="CAMI sample name (default: sample_0)")
parser.add_argument("--project-root", default=str(SCARB_ROOT), help="SCARB cluster data root")


def load_target_reads(sample, project_root):
    """read_names on a contig with zero CDS annotations at all, plus how many such contigs
    there are (for context in the printed report)."""
    contigs_path = Path(project_root) / "data" / "processed_data" / "CAMI_metagenome_no_cds_contigs" / f"{sample}_contigs_without_cds.txt"
    if not contigs_path.exists():
        raise FileNotFoundError(
            f"{contigs_path} not found - run cami_contigs_without_cds.py --sample {sample} first."
        )
    no_cds_contigs = set(contigs_path.read_text().split())

    tsv_path = Path(project_root) / "data" / "processed_data" / "reads_processed" / "test" / "CAMI_metagenome" / f"{sample}_cds_labels.tsv.gz"
    df = pd.read_csv(tsv_path, sep="\t", compression="gzip", usecols=["read_name", "contig_accession"])
    target_reads = set(df.loc[df["contig_accession"].isin(no_cds_contigs), "read_name"])
    return target_reads, len(no_cds_contigs)


def report(name, pkl_path, target_reads):
    if not Path(pkl_path).exists():
        print(f"  [skip] {name}: not found ({pkl_path})")
        return
    with open(pkl_path, "rb") as f:
        preds = pickle.load(f)
    hits = {r: d for r, d in preds.items() if r in target_reads}
    n_reads_with_pred = len(hits)
    n_cds_calls = sum(len(d["cds_coords"]) for d in hits.values())
    pct = 100 * n_reads_with_pred / len(target_reads) if target_reads else 0.0
    print(f"  {name}: {n_reads_with_pred:,} / {len(target_reads):,} target reads have >=1 predicted CDS "
          f"({pct:.3f}%), {n_cds_calls:,} total CDS calls")


def main():
    args = parser.parse_args()
    project_root = args.project_root

    print(f"Loading no-CDS-contig reads for {args.sample} ...")
    target_reads, n_contigs = load_target_reads(args.sample, project_root)
    print(f"{len(target_reads):,} reads on {n_contigs:,} contigs with zero CDS annotations.\n")

    print("Prodigal:")
    report(
        "Prodigal",
        f"{project_root}/data/processed_data/predictions/processed_predictions/prodigal_preds/CAMI_metagenome/{args.sample}/model_preds_dict.pkl",
        target_reads,
    )

    print("\nFragGeneScanRs:")
    fgs_root = f"{project_root}/data/processed_data/predictions/processed_predictions/fgs_preds"
    fgs_dirs = sorted(d for d in glob.glob(f"{fgs_root}/CAMI_metagenome_*") if Path(d).is_dir())
    for fgs_dir in fgs_dirs:
        error_model = Path(fgs_dir).name.replace("CAMI_metagenome_", "")
        report(f"FGS ({error_model})", f"{fgs_dir}/{args.sample}.pkl", target_reads)

    print("\nDeepCDS:")
    deepcds_glob = f"{project_root}/data/processed_data/predictions/processed_predictions/DeepCDS/*/CAMI_metagenome/*/{args.sample}/model_preds_dict.pkl"
    deepcds_paths = sorted(glob.glob(deepcds_glob))
    for p in deepcds_paths:
        parts = Path(p).parts
        # .../DeepCDS/{model_dir_path_suffix}/CAMI_metagenome/{model_preds_path}/{sample}/model_preds_dict.pkl
        model_preds_path = parts[-3]
        model_dir_path_suffix = parts[-5]
        report(f"DeepCDS/{model_dir_path_suffix} ({model_preds_path})", p, target_reads)


if __name__ == "__main__":
    main()
