"""
Postprocess MetaGeneMark-2 predictions on the read-based test sets into the same
model_preds_dict.pkl format postprocess_prodigal_preds.py produces.

Conventions are identical to the Prodigal postprocessing:
  - reads are oriented to the template strand, so only '+' strand CDS are kept
  - coordinates are 1-based inclusive on the read; rf = (start - 1) % 3
  - only predictions longer than 60 bp are kept (end - start + 1 > 60)
  - read_name = FASTA header up to the first "|"
Differences: comment lines are skipped explicitly (postprocess_prodigal_preds.py drops the
first line of the file instead, which there is a real CDS record), and the raw GFF3 files
are gzipped.

MetaGeneMark-2 CDS coordinates include the stop codon, like the test labels (checked on
GCF_000005845.2, error-free 300bp: GFF3 and GTF output identical; every near-miss has the
correct stop and frame).

Usage (called per test set by predict/predict_with_metagenemark2.sh):
  python postprocess_metagenemark2_preds.py --raw_dir <dir with {acc}.gff3[.gz]> --out_dir <dir> \
      --accessions <file>
Writes <out_dir>/{acc}/model_preds_dict.pkl
"""
import argparse
import gzip
import os
import pickle


def process_model_preds(gff3_path):
    """Parse a MetaGeneMark-2 GFF3 file into {read_name: {"cds_coords", "cds_fragments_connection"}}."""
    opener = gzip.open if gff3_path.endswith(".gz") else open
    model_dict = dict()
    with opener(gff3_path, "rt") as file:
        for line in file:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if fields[2] != "CDS" or fields[6] != "+":
                continue
            # MetaGeneMark-2 truncates the FASTA header at the first space; the read name
            # (everything before the first "|") is unaffected.
            read_name = fields[0].split("|")[0]
            cds_start, cds_end = int(fields[3]), int(fields[4])
            if cds_end - cds_start + 1 <= 60:
                continue
            entry = model_dict.setdefault(read_name, {"cds_coords": [], "cds_fragments_connection": []})
            entry["cds_fragments_connection"].append([len(entry["cds_coords"])])
            entry["cds_coords"].append([cds_start, cds_end, str((cds_start - 1) % 3)])
    return model_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--accessions", required=True)
    args = parser.parse_args()

    with open(args.accessions) as f:
        accessions = [line.strip() for line in f if line.strip()]

    n_cds = 0
    for acc in accessions:
        path = f"{args.raw_dir}/{acc}.gff3"
        if not os.path.exists(path):
            path += ".gz"
        preds = process_model_preds(path)
        n_cds += sum(len(v["cds_coords"]) for v in preds.values())
        os.makedirs(f"{args.out_dir}/{acc}", exist_ok=True)
        with open(f"{args.out_dir}/{acc}/model_preds_dict.pkl", "wb") as f:
            pickle.dump(preds, f)
    print(f"{len(accessions)} genomes, {n_cds:,} '+'-strand CDS predictions > 60 bp", flush=True)
