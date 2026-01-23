"""
Preprocessing script for DeepCDS Predictor training data.

This script reads the raw CSV files and converts them to memory-mapped numpy files
for efficient loading during training.

Usage:
    python preprocess_data.py --dataset_size 100_genomes --error_type indel_substitution
    python preprocess_data.py --dataset_size 200_genomes --error_type substitution
    python preprocess_data.py --dataset_size all_genomes --error_type none
"""

import numpy as np
import pandas as pd
import gc
import os
import pickle
import argparse
from transformers import AutoTokenizer

# Parse arguments
parser = argparse.ArgumentParser(description="Preprocess CDS Predictor training data")
parser.add_argument(
    "--dataset_size",
    type=str,
    default="100_genomes",
    choices=["100_genomes", "200_genomes", "400_genomes", "all_genomes"],
    help="Dataset size to preprocess (default: 100_genomes)",
)
parser.add_argument(
    "--error_type",
    type=str,
    default="indel_substitution",
    choices=["indel_substitution", "substitution", "none"],
    help="Type of data errors (default: indel_substitution)",
)
parser.add_argument(
    "--healthtech_cluster",
    type=bool,
    default=False,
    help="Whether running on HealthTech cluster (default: False)",
)
parser.add_argument(
    "--scarb_cluster",
    type=bool,
    default=False,
    help="Whether running on SCARB cluster (default: False)",
)
parser.add_argument(
    "--chunk_size",
    type=int,
    default=10000,
    help="Number of samples to process at a time (default: 10000)",
)
parser.add_argument(
    "--frac_val",
    type=float,
    default=0.25,
    help="Fraction of validation set to use, will follow original distribution of genomes and sequence types (default: 0.25)",
)
parser.add_argument(
    "--skip_validation",
    action="store_true",
    help="Skip validation set preprocessing (same validation set used for all training set sizes: use when validation already preprocessed for a certain error_type)",
)

args = parser.parse_args()

# Set paths based on error type
if args.error_type == "indel_substitution":
    model_dir_path_suffix = "model_with_errors"
elif args.error_type == "substitution":
    model_dir_path_suffix = "model_with_substitution_errors"
else:
    model_dir_path_suffix = "model_without_errors"

# Set input directory based on cluster
if args.healthtech_cluster:
    input_data_dir_path = f"../../../data/processed_data/model_data/shared_crf/{model_dir_path_suffix}"
elif args.scarb_cluster:
    input_data_dir_path = f"/tmp/nrt204/FragmentPredictor/data/processed_data/model_data/shared_crf/{model_dir_path_suffix}"
else:
    input_data_dir_path = f"../../../data/processed_data/model_data/shared_crf/{model_dir_path_suffix}"

# Constants
max_aa_len = 100                # Maximum amino acid length for training model
max_len = max_aa_len + 2        # Add CLS and EOS tokens
max_nt_len = max_aa_len * 3     # Maximum nucleotide length


def one_hot_encode_to_codon_format(sequence, max_aa_len) -> np.ndarray:
    """
    One-hot encode nucleotide sequence directly to codon format.

    Args:
        sequence: Nucleotide sequence string
        max_aa_len: Maximum amino acid length (sequence will be padded/truncated)

    Returns:
        numpy array of shape (max_aa_len, 12) with one-hot encoded codons
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoding = np.zeros((max_aa_len, 12), dtype=np.float32)

    for codon_idx in range(max_aa_len):
        nt_start = codon_idx * 3
        for offset in range(3):
            nt_idx = nt_start + offset
            if nt_idx < len(sequence):
                char = sequence[nt_idx]
                if char in mapping:
                    encoding[codon_idx, offset * 4 + mapping[char]] = 1.0

    return encoding


def preprocess_and_save(df, output_dir, tokenizer, chunk_size) -> None:
    """
    Process a DataFrame and save to memory-mapped files.

    Args:
        df: pandas DataFrame with the training or validation data
        output_dir: Directory to save the memory-mapped files
        tokenizer: HuggingFace tokenizer
        chunk_size: Number of samples to process at a time
    """
    os.makedirs(output_dir, exist_ok=True)
    n_samples = len(df)

    # Create memory-mapped files for each component
    # Shape information saved separately for loading
    shapes = {
        "n_samples": n_samples,
        "max_aa_len": max_aa_len,
        "max_len": max_len,
    }

    # Initialize memory-mapped arrays
    mmap_files = {}
    for rf in ["rf0", "rf1", "rf2"]:
        mmap_files[f"nt_{rf}"] = np.memmap(
            f"{output_dir}/nt_encodings_{rf}.npy",
            dtype="float32", mode="w+", shape=(n_samples, max_aa_len, 12)
        )
        mmap_files[f"aa_input_ids_{rf}"] = np.memmap(
            f"{output_dir}/aa_input_ids_{rf}.npy",
            dtype="int64", mode="w+", shape=(n_samples, max_len)
        )
        mmap_files[f"aa_attention_{rf}"] = np.memmap(
            f"{output_dir}/aa_attention_mask_{rf}.npy",
            dtype="int64", mode="w+", shape=(n_samples, max_len)
        )
        mmap_files[f"labels_{rf}"] = np.memmap(
            f"{output_dir}/labels_{rf}.npy",
            dtype="int8", mode="w+", shape=(n_samples, max_aa_len)
        )

    mmap_files["label_encodings"] = np.memmap(
        f"{output_dir}/label_encodings.npy",
        dtype="int8", mode="w+", shape=(n_samples, max_len - 2)
    )

    # Process shared label encodings first
    if isinstance(df["label_encodings"].iloc[0], str):
        df["label_encodings"] = df["label_encodings"].apply(eval)

    for i, lbl in enumerate(df["label_encodings"]):
        arr = np.array(lbl, dtype=np.int8)
        length = min(len(arr), max_len - 2)
        mmap_files["label_encodings"][i, :length] = arr[:length]
        mmap_files["label_encodings"][i, length:] = -1

        if i % 100000 == 0 and i > 0:
            print(f"  Processed {i}/{n_samples} label encodings", flush=True)
            mmap_files["label_encodings"].flush()

    mmap_files["label_encodings"].flush()
    df["label_encodings"] = None  # Free memory
    gc.collect()

    # Process each reading frame
    for rf in ["rf0", "rf1", "rf2"]:
        print(f"\nProcessing {rf}.", flush=True)

        # Process RF-specific labels
        if isinstance(df[f"{rf}_labels"].iloc[0], str):
            df[f"{rf}_labels"] = df[f"{rf}_labels"].apply(eval)

        for i, lbl in enumerate(df[f"{rf}_labels"]):
            arr = np.array(lbl, dtype=np.int8)
            length = min(len(arr), max_aa_len)
            mmap_files[f"labels_{rf}"][i, :length] = arr[:length]
            mmap_files[f"labels_{rf}"][i, length:] = -1

        mmap_files[f"labels_{rf}"].flush()
        df[f"{rf}_labels"] = None
        gc.collect()

        # Process in chunks for nucleotide and amino acid sequences
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)

            # Nucleotide encoding
            for i in range(start_idx, end_idx):
                seq = df[f"{rf}_seq_nt"].iloc[i]
                padded = seq + "N" * (max_nt_len - len(seq)) if len(seq) < max_nt_len else seq[:max_nt_len]
                mmap_files[f"nt_{rf}"][i] = one_hot_encode_to_codon_format(padded, max_aa_len)

            # Amino acid tokenization
            aa_sequences = df[f"{rf}_seq_aa"].iloc[start_idx:end_idx].tolist()
            aa_encodings = tokenizer(
                aa_sequences,
                padding="max_length",
                max_length=max_len,
                truncation=True,
                return_tensors="np",
            )
            mmap_files[f"aa_input_ids_{rf}"][start_idx:end_idx] = aa_encodings["input_ids"]
            mmap_files[f"aa_attention_{rf}"][start_idx:end_idx] = aa_encodings["attention_mask"]

            # Flush to disk
            mmap_files[f"nt_{rf}"].flush()
            mmap_files[f"aa_input_ids_{rf}"].flush()
            mmap_files[f"aa_attention_{rf}"].flush()

            print(f"  Processed {end_idx}/{n_samples} samples for {rf}", flush=True)

            del aa_sequences, aa_encodings
            gc.collect()

        # Free DataFrame columns
        df[f"{rf}_seq_nt"] = None
        df[f"{rf}_seq_aa"] = None
        gc.collect()

    # Save sequence descriptions
    seq_descs = df["seq_desc"].tolist()
    with open(f"{output_dir}/seq_descs.pkl", "wb") as f:
        pickle.dump(seq_descs, f)

    # Save shapes for loading
    with open(f"{output_dir}/shapes.pkl", "wb") as f:
        pickle.dump(shapes, f)

    # Close all memory-mapped files
    for key in mmap_files:
        del mmap_files[key]
    gc.collect()


def main():
    print("=" * 60)
    print("DeepCDS Predictor Data Preprocessing for more memory-efficient training")
    print("=" * 60)
    print(f"Dataset size: {args.dataset_size}")
    print(f"Error type: {args.error_type}")
    print(f"Input directory: {input_data_dir_path}")
    print(f"Chunk size: {args.chunk_size}")
    print("=" * 60)

    # Initialize tokenizer (same for all ESM-2 models)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", do_lower_case=False)

    # Load training data
    train_csv_path = f"{input_data_dir_path}/datasets_model/train_{args.dataset_size}.csv.gz"
    train_output_dir = f"{input_data_dir_path}/datasets_model/preprocessed_train_{args.dataset_size}"
    train_df = pd.read_csv(train_csv_path, index_col=None, compression="gzip")

    # Process and save training data
    preprocess_and_save(train_df, train_output_dir, tokenizer, args.chunk_size)

    # Cleanup
    del train_df
    gc.collect()

    # Process validation data (skip if already done for this error_type)
    if not args.skip_validation:
        val_output_dir = f"{input_data_dir_path}/datasets_model/preprocessed_val"

        val_csv_path = f"{input_data_dir_path}/datasets_model/val.csv.gz"
        val_df = pd.read_csv(val_csv_path, index_col=None, compression="gzip")

        # Subsample validation set
        if args.frac_val < 1.0:
            val_df["accession_seq_desc_merged"] = val_df["accession"].astype(str) + "_" + val_df["seq_desc"].astype(str)
            val_df = val_df.groupby("accession_seq_desc_merged", group_keys=False).apply(
                lambda x: x.sample(frac=args.frac_val, random_state=42)
            )
            print(f"Validation set subsampled to {len(val_df)} validation samples", flush=True)

        # Process and save validation data
        preprocess_and_save(val_df, val_output_dir, tokenizer, args.chunk_size)

        # Cleanup
        del val_df
        gc.collect()

    print("Processing of data complete.", flush=True)

if __name__ == "__main__":
    main()
