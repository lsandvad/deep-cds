"""Recall on CAMI CDS fragments from hypothetical-protein annotations.

Test set: CAMI_metagenome_no_cds_contigs (sample_0) - same contig filter as the main CAMI evaluation
(CAMI_metagenome_no_ab_initio_no_cds_contigs) but WITHOUT the exclusion of reads overlapping
hypothetical-protein / ab-initio-prediction CDS. In CAMI these CDS are labeled as coding (they are only
tagged), so all tools were evaluated on them.

1. Each labeled CDS fragment (testset_dict.pkl after get_full_cds) is placed on its contig in the read's
   strand frame, exactly as process_and_map_CAMI_testset.py labels reads: strand-frame position =
   start_coord + read position - 1 (indels shift this by at most a few bp), with minus-strand CDS converted
   to the reverse-complement frame. The annotated CDS (cached GFF3, raw_data/CAMI_datasets/gff_cache) with
   the largest overlap defines its category:
     hypothetical            - product=hypothetical protein, not an ab initio prediction
     hypothetical_ab_initio  - product=hypothetical protein AND "ab initio prediction"     (not evaluated)
     ab_initio               - "ab initio prediction", not a hypothetical protein          (not evaluated)
     known_function          - neither, and at least one go_function (GO molecular function) term
     named_no_go_function    - neither, without a go_function term                         (not evaluated)
   (pseudogene / pseudo=true / partial=true / programmed-frameshift CDS are not labeled at all).
2. TP per fragment and model: find_matching_genes(..., overlap_criteria=1) from the CAMI notebook
   (../cami_overlap_audit/cami_benchmark_functions.py), on all predictions and labels of the read.

Output: $RESULTS/hypothetical_counts.tsv - model x genome_bin_id x domain x category x length class: n, tp
        ($WORK/fragments.tsv.gz: per-fragment table)
Usage: python 01_classify_and_score.py [--work /tmp/nrt204/hypothetical_audit_work] [--procs 64]
"""
import argparse
import bisect
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = "/tmp/nrt204/FragmentPredictor/data/processed_data/hypothetical_protein_audit"
sys.path.insert(0, os.path.join(HERE, "..", "cami_overlap_audit"))
from cami_benchmark_functions import find_matching_genes, get_full_cds  # noqa: E402

D = "/tmp/nrt204/FragmentPredictor/data"
VARIANT, SAMPLE, OVERLAP = "CAMI_metagenome_no_cds_contigs", "sample_0", 1
GFF_CACHE = f"{D}/raw_data/CAMI_datasets/gff_cache"
CKPT = "full_model_all_genomes_seed_42_trained_final_8M_no_dropout"
PP = f"{D}/processed_data/predictions/processed_predictions"
PRED_PATHS = {
    "fgs_complete": f"{PP}/fgs_preds/{VARIANT}_complete/{SAMPLE}.pkl",
    "fgs_illumina_5": f"{PP}/fgs_preds/{VARIANT}_illumina_5/{SAMPLE}.pkl",
    "fgs_illumina_10": f"{PP}/fgs_preds/{VARIANT}_illumina_10/{SAMPLE}.pkl",
    "prodigal": f"{PP}/prodigal_preds/{VARIANT}/{SAMPLE}/model_preds_dict.pkl",
    "deep_cds_no_errors": f"{PP}/DeepCDS/model_without_errors/{VARIANT}/{CKPT}/{SAMPLE}/model_preds_dict.pkl",
    "deep_cds_substitution": f"{PP}/DeepCDS/model_with_substitution_errors/{VARIANT}/{CKPT}/{SAMPLE}/model_preds_dict.pkl",
    "deep_cds_errors": f"{PP}/DeepCDS/model_with_errors/{VARIANT}/{CKPT}/{SAMPLE}/model_preds_dict.pkl",
}
CACHED_OVERALL = f"{D}/processed_data/report_results/{VARIANT}/cds_level_metagenome_model_comparison/oc_{OVERLAP}/results_overall.pkl"
EXCLUDE = ("pseudogene", "pseudo=true", "partial=true", "note=programmed frameshift")
LENGTH_BINS, LENGTH_LABELS = [60, 89, 119, 10_000], ["61-89 bp", "90-119 bp", ">=120 bp"]

testset_dict = None
keys = None


def parse_contig(contig):
    """{strand: (starts, ends, categories)} sorted by start, in the strand frame used for labeling."""
    path = f"{GFF_CACHE}/{contig}.gff3"
    if not os.path.exists(path):
        return contig, None
    length, raw = None, {"+": [], "-": []}
    with open(path) as fh:
        for line in fh:
            if line.startswith("##sequence-region"):
                length = int(line.split()[3])
                continue
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9 or c[2] != "CDS" or c[6] not in "+-":
                continue
            a = c[8].lower()
            if any(m in a for m in EXCLUDE):
                continue
            hyp, abi = "product=hypothetical protein" in a, "ab initio prediction" in a
            cat = ("hypothetical_ab_initio" if hyp and abi else "hypothetical" if hyp
                   else "ab_initio" if abi else "known_function" if "go_function=" in a
                   else "named_no_go_function")
            raw[c[6]].append((int(c[3]), int(c[4]), cat))
    out = {}
    for strand, iv in raw.items():
        if strand == "-":
            iv = [(length - e + 1, length - s + 1, cat) for s, e, cat in iv]
        iv.sort()
        out[strand] = ([s for s, _, _ in iv], [e for _, e, _ in iv], [cat for _, _, cat in iv])
    return contig, out


def classify(interval_index, s, e):
    starts, ends, cats = interval_index
    best, best_ov = "unmapped", 0
    # candidates: intervals starting before e; scan back while they could still overlap (CDS <= ~100 kb)
    i = bisect.bisect_right(starts, e) - 1
    while i >= 0 and starts[i] >= s - 100_000:
        ov = min(ends[i], e) - max(starts[i], s) + 1
        if ov > best_ov:
            best, best_ov = cats[i], ov
        i -= 1
    return best


def match_model(model):
    with open(PRED_PATHS[model], "rb") as f:
        preds = get_full_cds(pickle.load(f))
    pos = {k: i for i, k in enumerate(keys)}
    tp = np.zeros(len(keys), dtype=bool)
    for read, entry in testset_dict.items():
        pairs, _ = find_matching_genes(preds.get(read, {}).get("cds_coords", []), entry["cds_coords"], OVERLAP)
        for _, a in pairs:
            tp[pos[(read, a)]] = True
    print(f"  {model}: {int(tp.sum()):,} TP", flush=True)
    return model, tp


def main():
    global testset_dict, keys
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/hypothetical_audit_work")
    parser.add_argument("--procs", type=int, default=64)
    args = parser.parse_args()
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(f"{RESULTS}", exist_ok=True)

    with open(f"{D}/processed_data/testset_processed/{VARIANT}/{SAMPLE}/testset_dict.pkl", "rb") as f:
        testset_dict = get_full_cds(pickle.load(f))
    keys = [(r, i) for r, e in testset_dict.items() for i in range(len(e["cds_coords"]))]
    print(f"{len(testset_dict):,} reads with labeled CDS, {len(keys):,} labeled CDS fragments", flush=True)

    with Pool(7) as pool:
        tp_flags = dict(pool.map(match_model, list(PRED_PATHS)))

    meta = pd.read_csv(f"{D}/processed_data/reads_processed/test/CAMI_metagenome/{SAMPLE}_cds_labels.tsv.gz", sep="\t",
                       compression="gzip", usecols=["read_name", "start_coord", "contig_accession", "strand",
                                                    "genome_bin_id", "domain", "uncertain_region_overlap"])
    meta = meta[meta["read_name"].isin(testset_dict.keys())].set_index("read_name")
    contigs = sorted(meta["contig_accession"].unique())
    with Pool(args.procs) as pool:
        index = dict(pool.map(parse_contig, contigs, chunksize=4))
    print(f"{len(contigs):,} contigs; GFF missing for {sum(v is None for v in index.values())}", flush=True)

    md = dict(zip(meta.index, zip(meta["start_coord"], meta["contig_accession"], meta["strand"], meta["genome_bin_id"],
                                  meta["domain"], meta["uncertain_region_overlap"])))
    rows = []
    for read, i in keys:
        coord = testset_dict[read]["cds_coords"][i]
        start_coord, contig, strand, gbin, domain, tagged = md[read]
        s, e = int(start_coord) + coord[0] - 1, int(start_coord) + coord[1] - 1
        ix = index.get(contig)
        cat = classify(ix[strand], s, e) if ix else "unmapped"
        rows.append((read, i, gbin, domain, bool(tagged), cat, coord[1] - coord[0] + 1))
    df = pd.DataFrame(rows, columns=["read_name", "coord_idx", "genome_bin_id", "domain", "read_tagged", "category", "length_bp"])
    for model, tp in tp_flags.items():
        df[model] = tp

    df.to_csv(f"{args.work}/fragments.tsv.gz", sep="\t", index=False, compression="gzip")

    # consistency check: fragments assigned to a tagged CDS should lie on reads flagged uncertain_region_overlap
    tagged_cat = df["category"].isin(["hypothetical", "hypothetical_ab_initio", "ab_initio"])
    mismatch = tagged_cat & ~df["read_tagged"]
    print(f"fragments assigned to a tagged CDS: {int(tagged_cat.sum()):,}; of these on reads NOT flagged "
          f"uncertain_region_overlap: {int(mismatch.sum()):,}", flush=True)
    df[mismatch].head(200).to_csv(f"{args.work}/tag_mismatch_examples.tsv", sep="\t", index=False)
    if os.path.exists(CACHED_OVERALL):
        with open(CACHED_OVERALL, "rb") as f:
            cached = pickle.load(f)
        for model in PRED_PATHS:
            assert int(df[model].sum()) == int(cached[model]["tp"]), (model, int(df[model].sum()), cached[model]["tp"])
            assert len(df) == int(cached[model]["tp"] + cached[model]["fn"]), model
        print("Check passed: TP and labeled-CDS totals equal the CAMI notebook's cached results for all models")
    else:
        print(f"(no cached CAMI results at {CACHED_OVERALL} to cross-check against)")

    df["length_class"] = pd.cut(df["length_bp"], LENGTH_BINS, labels=LENGTH_LABELS).astype(str)
    long = df.melt(id_vars=["genome_bin_id", "domain", "category", "length_class"], value_vars=list(PRED_PATHS),
                   var_name="model", value_name="tp")
    agg = (long.groupby(["model", "genome_bin_id", "domain", "category", "length_class"], observed=True)["tp"]
           .agg(n="size", tp="sum").reset_index())
    agg.to_csv(f"{RESULTS}/hypothetical_counts.tsv", sep="\t", index=False)
    print(df["category"].value_counts().to_string())
    print("DONE")


if __name__ == "__main__":
    main()
