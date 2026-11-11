"""Biologically invalid transitions in the CRF of the trained DeepCDS models.

1. Learned transition scores: every entry of the CRF transition matrix with the group it was initialized in during
   training (train_full_model.py): invalid -> -10, valid but infrequent -> transition_weight (tuned per model),
   frequent self-transitions -> 0. A transition between two shared labels is valid if it is valid in all three
   reading frames (same rules as in training).
2. Decoded label sequences: the test reads are decoded as in the benchmark (benchmark/predict/predict_with_DeepCDS.py:
   each read as given, CRF Viterbi decoding over the codon positions), and transitions between consecutive decoded
   labels that are invalid are counted.

Outputs ($RESULTS/): learned_transitions_{model}.tsv, decoded_transitions_{model}_{data_dir}.tsv (per genome: reads,
reads with an invalid transition, transitions, invalid transitions) and invalid_transition_pairs_{model}_{data_dir}.tsv.
Usage: python crf_invalid_transitions.py --model N|S|SI --gpu 0 [--data_dir with_errors_5e-06i_0.004s_300bp]
"""
import argparse
import os
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src_dev.modeling import encode_data, load_model  # noqa: E402

DATA = "/tmp/nrt204/FragmentPredictor/data/processed_data"
MODELS = {"N": ("model_without_errors", 4), "S": ("model_with_substitution_errors", 4), "SI": ("model_with_errors", 6)}
CKPT = "full_model_all_genomes_seed_42_trained_final_8M_no_dropout.pth"
ESM2 = "facebook/esm2_t6_8M_UR50D"
# per-frame label -> allowed next labels, as in training
# 0 non-coding, 1 CDS, 2 start codon, 3 stop codon, 4/5 transition into/out of a CDS caused by an indel error
LEGAL = {4: {0: {0, 2}, 1: {1, 3}, 2: {1}, 3: {0, 2}},
         6: {0: {0, 2, 4}, 1: {1, 3, 5}, 2: {1, 5}, 3: {0, 2, 4}, 4: {1, 3, 5}, 5: {0, 2, 4}}}
FREQUENT = {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)}


def validity_mask(mapping, n_classes):
    """valid[i, j]: transition from shared label i to j is legal in all three reading frames."""
    k = len(mapping)
    return np.array([[all(mapping[j][f] in LEGAL[n_classes][mapping[i][f]] for f in range(3)) for j in range(k)]
                     for i in range(k)])


def stage(accessions, data_dir, work):
    """Copy the test read CSVs to local disk in parallel (the ERDA mount is slow per file)."""
    os.makedirs(f"{work}/{data_dir}", exist_ok=True)
    def copy(acc):
        dst = f"{work}/{data_dir}/{acc}.csv.gz"
        if not os.path.exists(dst):
            part = f"{dst}.{os.getpid()}.part"   # per process: several models may stage at the same time
            shutil.copyfile(f"{DATA}/reads_processed/test/{data_dir}/csv/{acc}.csv.gz", part)
            os.replace(part, dst)
    with ThreadPoolExecutor(32) as pool:
        list(pool.map(copy, accessions))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_dir", default="with_errors_5e-06i_0.004s_300bp")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_genomes", type=int, default=None, help="for testing")
    parser.add_argument("--work", default="/tmp/nrt204/model_behavior_work")
    parser.add_argument("--results", default=f"{DATA}/model_behavior/crf_transitions")
    args = parser.parse_args()
    os.makedirs(args.results, exist_ok=True)
    model_dir, n_classes = MODELS[args.model]
    device = torch.device(f"cuda:{args.gpu}")

    model, mapping = load_model(CKPT, f"{DATA}/model_data/shared_crf/{model_dir}", device=device, esm2_model=ESM2,
                                label_classes=n_classes)
    model.eval()
    valid = validity_mask(mapping, n_classes)
    frequent = np.zeros_like(valid)
    for i, labels in mapping.items():
        frequent[i, i] = tuple(labels) in FREQUENT

    # 1. learned transition scores by initialization group
    learned = model.CRF.crf.transitions.detach().cpu().numpy()
    group = np.where(~valid, "invalid", np.where(frequent, "frequent", "valid_infrequent"))
    i, j = np.indices(learned.shape)
    pd.DataFrame({"from_label": i.ravel(), "to_label": j.ravel(),
                  "from_rf": [mapping[x] for x in i.ravel()], "to_rf": [mapping[x] for x in j.ravel()],
                  "init_group": group.ravel(), "learned_score": learned.ravel()}
                 ).to_csv(f"{args.results}/learned_transitions_{args.model}.tsv", sep="\t", index=False)

    # 2. invalid transitions in the decoded label sequences (genomes using translation table 11)
    with open(f"{DATA}/dataset_information/table4_accessions") as f:
        table4 = set(re.findall(r'"(GCF_\d+\.\d+)"', f.read()))
    with open(f"{DATA}/genome_partitions/test_partition_accessions.txt") as f:
        accessions = [a for a in f.read().split() if a not in table4][:args.max_genomes]
    stage(accessions, args.data_dir, args.work)
    tokenizer = AutoTokenizer.from_pretrained(ESM2, do_lower_case=False)
    max_aa_len = int(np.ceil(int(args.data_dir.split("_")[-1].rstrip("bp")) / 3)) + 5   # as in the benchmark
    invalid_t = torch.tensor(~valid)

    rows, pairs = [], Counter()
    for n, acc in enumerate(accessions):
        reads = pd.read_csv(f"{args.work}/{args.data_dir}/{acc}.csv.gz")
        loader = DataLoader(encode_data(reads, max_aa_len, tokenizer), batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
        n_reads = n_bad_reads = n_trans = n_bad = 0
        with torch.inference_mode():
            for batch in loader:
                inputs = []
                for rf in ("rf0", "rf1", "rf2"):
                    inputs += [batch[f"nt_encodings_{rf}"].to(device), batch[f"aa_encodings_{rf}"]["input_ids"].to(device),
                               batch[f"aa_encodings_{rf}"]["attention_mask"].to(device)]
                for path in model(*inputs)["predictions"]:
                    path = torch.tensor(path)
                    bad = invalid_t[path[:-1], path[1:]]
                    n_reads += 1
                    n_trans += len(path) - 1
                    if bad.any():
                        n_bad_reads += 1
                        n_bad += int(bad.sum())
                        pairs.update(zip(path[:-1][bad].tolist(), path[1:][bad].tolist()))
        rows.append({"genome": acc, "reads": n_reads, "reads_with_invalid": n_bad_reads, "transitions": n_trans,
                     "invalid_transitions": n_bad})
        print(f"{n + 1}/{len(accessions)} {acc}: {n_reads:,} reads, {n_bad_reads} with an invalid transition", flush=True)

    tag = f"{args.model}_{args.data_dir}"
    pd.DataFrame(rows).to_csv(f"{args.results}/decoded_transitions_{tag}.tsv", sep="\t", index=False)
    pd.DataFrame([{"from_label": a, "to_label": b, "from_rf": mapping[a], "to_rf": mapping[b], "count": c}
                  for (a, b), c in pairs.most_common()],
                 columns=["from_label", "to_label", "from_rf", "to_rf", "count"]
                 ).to_csv(f"{args.results}/invalid_transition_pairs_{tag}.tsv", sep="\t", index=False)
    print("DONE")


if __name__ == "__main__":
    main()
