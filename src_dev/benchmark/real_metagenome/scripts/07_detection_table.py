"""Step 2 postprocessing: per read (>= 61 bp) and tool, is at least one CDS >= 61 bp predicted (either strand)?

  Prodigal / FGS: CDS lines of the GFF (both strands are searched by the tools themselves).
  DeepCDS: CDS lines of predictions_reads_all.gff (all reads; those < 61 bp are skipped here) for the read (<id>, + strand) and its reverse complement
           (<id>_rc). Segments of one CDS split at a predicted indel share a group_id (<group>.<k>); they are
           merged to their full extent before the length filter, as get_full_cds does in the benchmark.
  CDS < 61 bp are discarded for every tool.
Output: WORK/detection/detection.tsv.gz - read_id + one 0/1 column per tool, in reads_ge61.fa order
Usage: python 07_detection_table.py [--work ...] [--deepcds_dir <dir holding the DeepCDS GFFs>]
"""
import argparse
import glob
import gzip
import os
import subprocess

import numpy as np
import pandas as pd

MIN_CDS = 61
CKPT = "full_model_all_genomes_seed_42_trained_final_8M_no_dropout"
DEEPCDS = {"deepcds_N": "model_without_errors", "deepcds_S": "model_with_substitution_errors", "deepcds_SI": "model_with_errors"}
TOOLS = ["deepcds_N", "deepcds_S", "deepcds_SI", "prodigal", "fgs_complete", "fgs_illumina_5", "fgs_illumina_10"]


def read_ids(work):
    proc = subprocess.Popen(f"pigz -dc {work}/fasta/reads_ge61.fa.gz | grep '^>' | cut -c2-",
                            shell=True, stdout=subprocess.PIPE, text=True)
    ids = [line.rstrip("\n") for line in proc.stdout]
    proc.wait()
    return ids


def detected_from_gff(paths, index, deepcds=False):
    hit = np.zeros(len(index), dtype=bool)
    n_cds = n_kept = 0
    for path in paths:
        groups = {}
        with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                c = line.split("\t", 9)
                if len(c) < 9 or c[2] != "CDS":
                    continue
                read, s, e = c[0], int(c[3]), int(c[4])
                if deepcds:
                    read = read[:-3] if read.endswith("_rc") else read
                    gid = next((a[9:].rsplit(".", 1)[0] for a in c[8].split(";") if a.startswith("group_id=")), None)
                    if gid is not None:
                        key = (c[0], gid)
                        gs, ge = groups.get(key, (s, e))
                        groups[key] = (min(gs, s), max(ge, e))
                        continue
                if read not in index:
                    continue  # read < 61 bp (DeepCDS input holds all reads)
                n_cds += 1
                if e - s + 1 >= MIN_CDS:
                    n_kept += 1
                    hit[index[read]] = True
        for (seqid, _), (s, e) in groups.items():
            read = seqid[:-3] if seqid.endswith("_rc") else seqid
            if read not in index:
                continue
            n_cds += 1
            if e - s + 1 >= MIN_CDS:
                n_kept += 1
                hit[index[read]] = True
    return hit, n_cds, n_kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/real_metagenome_poc_work")
    parser.add_argument("--deepcds_dir", default="/tmp/nrt204/real_metagenome_poc_work/preds/DeepCDS")
    args = parser.parse_args()
    ids = read_ids(args.work)
    index = {r: i for i, r in enumerate(ids)}
    print(f"{len(ids):,} reads >= {MIN_CDS} bp", flush=True)

    det = pd.DataFrame({"read_id": ids})
    log = []
    for tool in TOOLS:
        if tool in DEEPCDS:
            paths = [f"{args.deepcds_dir}/{DEEPCDS[tool]}/real_metagenome_poc/{CKPT}/predictions_reads_all.gff"]
        else:
            paths = sorted(glob.glob(f"{args.work}/preds/{tool}/chunk_*.gff.gz"))
        missing = [p for p in paths if not os.path.exists(p)]
        if not paths or missing:
            print(f"  {tool}: predictions missing, skipped ({missing or 'no files'})", flush=True)
            continue
        hit, n_cds, n_kept = detected_from_gff(paths, index, deepcds=tool in DEEPCDS)
        det[tool] = hit.astype(np.uint8)
        log.append((tool, len(paths), n_cds, n_kept, int(hit.sum())))
        print(f"  {tool}: {n_cds:,} CDS, {n_kept:,} >= {MIN_CDS} bp, {int(hit.sum()):,} reads detected", flush=True)

    os.makedirs(f"{args.work}/detection", exist_ok=True)
    det.to_csv(f"{args.work}/detection/detection.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(log, columns=["tool", "gff_files", "cds", f"cds_ge{MIN_CDS}", "reads_detected"]).to_csv(
        f"{args.work}/detection/prediction_counts.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
