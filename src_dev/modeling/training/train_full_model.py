import gc
import json
import os
import pickle
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
from sklearn.metrics import matthews_corrcoef
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer

torch.cuda.empty_cache()  # Clear the GPU memory cache
pd.options.mode.chained_assignment = None

import argparse

# Add argument parser at the beginning
parser = argparse.ArgumentParser(description="Train DeepCDS Model")
parser.add_argument(
    "--dataset_size",
    type=str,
    default="100_genomes",
    choices=["100_genomes", "200_genomes", "400_genomes", "all_genomes"],
    help="Dataset size to use for training (default: 100_genomes)",
)
parser.add_argument("--gpu", type=int, default=0, help="GPU number to use (default: 0)")
parser.add_argument("--healthtech_cluster", action="store_true", help="Whether running on HealthTech cluster")
parser.add_argument("--scarb_cluster", action="store_true", help="Whether running on SCARB cluster")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
parser.add_argument(
    "--error_type",
    type=str,
    default="indel_substitution",
    choices=["indel_substitution", "substitution", "none"],
    help="Type of data errors to include (default: indel_substitution)",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume training from the latest checkpoint",
)
parser.add_argument(
    "--esm_model",
    type=str,
    default="8M",
    choices=["8M", "35M", "150M", "650M"],
    help="ESM-2 model size to use (default: 8M)",
)

args = parser.parse_args()

# Set variables based on error type
if args.error_type == "indel_substitution":
    model_dir_path_suffix = "model_with_errors"
    label_classes = 6
    wandb_project_name = "DeepCDS_errors"
    steps_between_vals = 10000
elif args.error_type == "substitution":
    model_dir_path_suffix = "model_with_substitution_errors"
    label_classes = 4
    wandb_project_name = "DeepCDS_substitution_errors"
    steps_between_vals = 6000
else:
    model_dir_path_suffix = "model_without_errors"
    label_classes = 4
    wandb_project_name = "DeepCDS_no_errors"
    steps_between_vals = 5000


debug_mode = False  # Set to True for debugging with smaller dataset

if debug_mode == True:
    steps_between_vals = 5000
    frac_train = 0.005
    frac_val = 0.0025 
    model_checkpoint_extension = "_debug"
    wandb_project_name = "debug"

else: 
    frac_train = 1.0
    frac_val = 0.25
    model_checkpoint_extension = "_final"

# Configure CUDA memory allocations.
# expandable_segments:True (PyTorch 2.0+) eliminates allocator fragmentation by using
# memory-mapped regions that can grow/shrink on demand, instead of fixed-size blocks.
# This replaces the old max_split_size_mb:128 setting which was causing ~2.4 GB of
# fragmentation to accumulate per backward pass (fp16 forward blocks vs fp32 backward
# gradient buffers have different sizes and cannot reuse each other's cache slots).
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if args.healthtech_cluster:
    input_data_dir_path = f"/home/projects/DeepCDStmp/data/processed_data/model_data/{model_dir_path_suffix}"
    num_workers_cpu = 2
    pin_memory = True
    # Use argparse values
    device = torch.device(f"cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device_type = device.type  # "cuda", "mps", or "cpu"

    print("Device: ", device, flush=True)

    assert device == torch.device("cuda"), "HealthTech cluster run should be on a CUDA GPU."
    os.makedirs(f"{input_data_dir_path}/models/", exist_ok=True)
    models_output_dir_path = f"{input_data_dir_path}/models/"

elif args.scarb_cluster:
    input_data_dir_path = f"/tmp/nrt204/FragmentPredictor/data/processed_data/model_data/shared_crf/{model_dir_path_suffix}"  # SCARB cluster
    num_workers_cpu = 2
    pin_memory = True
    # Use argparse values
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device_type = device.type  # "cuda", "mps", or "cpu"
    print("Device: ", device, flush=True)

    assert device.type == "cuda", "SCARB cluster run should be on a CUDA GPU."
    print(f"Device type: {device_type}, GPU: {args.gpu if device_type == 'cuda' else 'N/A'}", flush=True)

    # Cap PyTorch's CUDA memory usage to avoid starving other GPU users.
    # Scarb GPUs have 80 GB; we claim 48 GB, leaving ~32 GB for others.
    _cuda_mem_limit_gb = 48
    _total_vram = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(_cuda_mem_limit_gb * 1e9 / _total_vram, device)
    print(f"CUDA memory capped at {_cuda_mem_limit_gb} GB / {_total_vram / 1e9:.1f} GB total", flush=True)

    os.makedirs(f"{input_data_dir_path}/models/", exist_ok=True)
    models_output_dir_path = f"{input_data_dir_path}/models/"


else:
    # Use argparse values
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device_type = device.type  # "cuda", "mps", or "cpu"

    input_data_dir_path = f"../../../data/processed_data/model_data/shared_crf/{model_dir_path_suffix}"
    num_workers_cpu = 0
    pin_memory = False

    print(f"Device type: {device_type}, GPU: {args.gpu if device_type == 'cuda' else 'N/A'}", flush=True)

    os.makedirs(f"{input_data_dir_path}/models/", exist_ok=True)
    models_output_dir_path = f"{input_data_dir_path}/models/"

# ESM model choice based on argument
if args.esm_model == "650M":
    esm2_model = "facebook/esm2_t33_650M_UR50D"
    # Recommend gradient checkpointing for 650M model
    gradient_checkpointing = True
    # Smaller batch sizes for larger model
    default_train_batch_size = 32
    default_val_batch_size = 64
elif args.esm_model == "150M":
    esm2_model = "facebook/esm2_t30_150M_UR50D"
    # Recommend gradient checkpointing for 150M model
    gradient_checkpointing = True
    # Smaller batch sizes for larger model
    default_train_batch_size = 32
    default_val_batch_size = 128
elif args.esm_model == "35M":
    esm2_model = "facebook/esm2_t12_35M_UR50D"
    gradient_checkpointing = True
    # Smaller batch sizes for larger model
    default_train_batch_size = 32
    default_val_batch_size = 64
else:
    esm2_model = "facebook/esm2_t6_8M_UR50D"
    default_train_batch_size = 32
    default_val_batch_size = 450
    gradient_checkpointing = False

print(f"Using ESM-2 model: {esm2_model}", flush=True)
print(f"Gradient checkpointing (used by default for 35M, 150M and 650M versions of ESM-2): {gradient_checkpointing}", flush=True)

dataset_size = args.dataset_size  # Use argparse value

max_aa_len = 100
max_len = max_aa_len + 2  # Add CLS and EOS tokens

######################################################################################################################################################################################################
############################################################################################Define functions##########################################################################################
######################################################################################################################################################################################################


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Seed function for DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def clear_memory():
    """
    Clear GPU and CPU memory caches to free up resources.
    Empties CUDA cache if GPU is available and triggers garbage collection.
    """
    # Memory clean up function
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    step,
    val_times_counter,
    best_val_loss,
    counter_patience,
    warmup_state,
    esm2_initial_params=None,
    wandb_run_id=None,
):
    """
    Save a full training checkpoint for resumption.

    Args:
        checkpoint_path: Path to save the checkpoint
        model: The model being trained
        optimizer: The optimizer
        scheduler: The learning rate scheduler
        scaler: The gradient scaler for mixed precision (can be None)
        epoch: Current epoch number
        step: Current global step
        val_times_counter: Number of validation runs completed
        best_val_loss: Best validation loss seen so far
        counter_patience: Current patience counter for early stopping
        warmup_state: Warmup state dict (can be None)
        esm2_initial_params: Initial ESM-2 parameters for tracking changes (can be None)
        wandb_run_id: WandB run ID for resuming the same run (can be None)
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "val_times_counter": val_times_counter,
        "best_val_loss": best_val_loss,
        "counter_patience": counter_patience,
        "warmup_state": warmup_state,
        "esm2_initial_params": esm2_initial_params,
        "wandb_run_id": wandb_run_id,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}", flush=True)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device):
    """
    Load a training checkpoint and restore all training state.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load weights into
        optimizer: The optimizer to restore state
        scheduler: The learning rate scheduler to restore state
        scaler: The gradient scaler (can be None)
        device: The device to load tensors to

    Returns:
        dict: Dictionary containing restored training state variables
    """
    print(f"Loading checkpoint from {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Check if ESM-2 layers were unfrozen when checkpoint was saved
    # by comparing optimizer param group counts
    saved_param_groups = len(checkpoint["optimizer_state_dict"]["param_groups"])
    current_param_groups = len(optimizer.param_groups)

    if saved_param_groups > current_param_groups:
        # ESM-2 was unfrozen - need to add those params to optimizer first
        print(f"Checkpoint has {saved_param_groups} param groups, current has {current_param_groups}", flush=True)
        print("Adding ESM-2 parameters to optimizer before loading state...", flush=True)

        # Unfreeze ESM-2 layers and add to optimizer
        unfreeze_start = model.sequence_encoder.unfreeze_start
        if unfreeze_start is not None:
            for i, layer in enumerate(model.sequence_encoder.pretrained_model_aa.encoder.layer):
                if i >= unfreeze_start:
                    for param in layer.parameters():
                        param.requires_grad = True

            # Collect and add ESM-2 params to optimizer (with dummy LR, will be overwritten)
            unfrozen_params = [
                p for layer in model.sequence_encoder.pretrained_model_aa.encoder.layer[unfreeze_start:]
                for p in layer.parameters()
            ]
            optimizer.add_param_group({"params": unfrozen_params, "lr": 1e-5})

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Use .get() with defaults for fields that might not exist in older checkpoints
    restored_state = {
        "epoch": checkpoint["epoch"],
        "step": checkpoint["step"],
        "val_times_counter": checkpoint["val_times_counter"],
        "best_val_loss": checkpoint["best_val_loss"],
        "counter_patience": checkpoint["counter_patience"],
        "warmup_state": checkpoint.get("warmup_state"),
        "esm2_initial_params": checkpoint.get("esm2_initial_params"),
    }

    print(f"Resumed from epoch {restored_state['epoch']}, step {restored_state['step']}", flush=True)
    print(f"Best val loss so far: {restored_state['best_val_loss']:.4f}", flush=True)

    return restored_state


def one_hot_encode(sequence):
    """
    One-hot encode nucleotide sequences in a matrix format of 4 rows (A, C, G, T)
    and len(sequence) columns.

    Args:
        sequence (str): a nucleotide sequence.

    Returns:
        encoding (tensor): one-hot encoded sequence.
    """

    # Define the mapping of nucleotides to indices
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    # Create an empty one-hot encoding tensor
    encoding = torch.zeros(4, len(sequence))

    # Convert the sequence to a tensor of indices for efficient indexing
    indices = torch.tensor([mapping[char] for char in sequence if char in mapping], dtype=torch.long)

    # Use advanced indexing to set the appropriate positions to 1
    positions = torch.arange(len(sequence))[[char in mapping for char in sequence]]
    encoding[indices, positions] = 1

    # For 'N', we do nothing, so the corresponding column remains all zeros

    return torch.tensor(encoding)


def process_nt_sequences_to_codons(nt_sequences, max_aa_len):
    """
    Convert nucleotide sequences from dimension (4, nucleotide_seq_len) to dimension (12, nucleotide_seq_len/3) format
    by grouping every 3 nucleotides (1 codon) together.

    Args:
        nt_sequences: List of tensors with shape (4, nucleotide_seq_len)

    Returns:
        List of tensors with shape (12, nucleotide_seq_len/3)
    """
    processed_sequences = []

    for seq in nt_sequences:
        # seq has shape (4, nucleotide_seq_len) -> reshape to (4, nucleotide_seq_len/3, 3) to group every 3 nucleotides
        seq_reshaped = seq.view(4, max_aa_len, 3)

        # Transpose to (nucleotide_seq_len/3, 4, 3) then reshape to (nucleotide_seq_len/3, 12)
        seq_transposed = seq_reshaped.transpose(0, 1)  # (nucleotide_seq_len/3, 4, 3)
        seq_formatted = seq_transposed.reshape(max_aa_len, 12)  # (nucleotide_seq_len/3, 12)

        processed_sequences.append(seq_formatted)

    return processed_sequences


class SeqDataset(torch.utils.data.Dataset):
    """
    Dataset class for multi-reading-frame sequence data.

    Stores nucleotide encodings, amino acid encodings, and labels for all three
    reading frames (rf0, rf1, rf2) along with sequence descriptions.

    Args:
        nt_encodings_rf0 (list): List of nucleotide codon encodings (max_aa_len, 12) for reading frame 0
        aa_encodings_rf0 (BatchEncoding): Tokenized amino acid encodings (dict) for reading frame 0
        labels_rf0 (list): List of numpy arrays (int8) with per-position labels for reading frame 0
        nt_encodings_rf1 (list): List of nucleotide codon encodings (max_aa_len, 12) for reading frame 1
        aa_encodings_rf1 (BatchEncoding): Tokenized amino acid encodings (dict) for reading frame 1
        labels_rf1 (list): List of numpy arrays (int8) with per-position labels for reading frame 1
        nt_encodings_rf2 (list): List of nucleotide codon encodings (max_aa_len, 12) for reading frame 2
        aa_encodings_rf2 (BatchEncoding): Tokenized amino acid encodings (dict) for reading frame 2
        labels_rf2 (list): List of numpy arrays (int8) with per-position labels for reading frame 2
        label_encodings (np.ndarray): Padded array of shape (num_samples, max_len-2) with mapped label sequences (int8)
        seq_desc (list): List of sequence description strings/identifiers

    Returns:
        Dictionary containing all encodings and labels for a single sequence across
        all three reading frames, with tensors converted to appropriate dtypes.
    """

    def __init__(
        self,
        nt_encodings_rf0,
        aa_encodings_rf0,
        labels_rf0,
        nt_encodings_rf1,
        aa_encodings_rf1,
        labels_rf1,
        nt_encodings_rf2,
        aa_encodings_rf2,
        labels_rf2,
        label_encodings,
        seq_desc,
    ):
        self.nt_encodings_rf0 = nt_encodings_rf0
        self.aa_encodings_rf0 = aa_encodings_rf0
        self.labels_rf0 = labels_rf0

        self.nt_encodings_rf1 = nt_encodings_rf1
        self.aa_encodings_rf1 = aa_encodings_rf1
        self.labels_rf1 = labels_rf1

        self.nt_encodings_rf2 = nt_encodings_rf2
        self.aa_encodings_rf2 = aa_encodings_rf2
        self.labels_rf2 = labels_rf2

        self.label_encodings = label_encodings
        self.seq_desc = seq_desc

    def __getitem__(self, idx):
        item = {
            "nt_encodings_rf0": torch.as_tensor(self.nt_encodings_rf0[idx], dtype=torch.float32),
            "aa_encodings_rf0": {key: val[idx] for key, val in self.aa_encodings_rf0.items()},
            "labels_rf0": torch.from_numpy(self.labels_rf0[idx]) if isinstance(self.labels_rf0[idx], np.ndarray) else torch.tensor(self.labels_rf0[idx], dtype=torch.float32),
            "nt_encodings_rf1": torch.as_tensor(self.nt_encodings_rf1[idx], dtype=torch.float32),
            "aa_encodings_rf1": {key: val[idx] for key, val in self.aa_encodings_rf1.items()},
            "labels_rf1": torch.from_numpy(self.labels_rf1[idx]) if isinstance(self.labels_rf1[idx], np.ndarray) else torch.tensor(self.labels_rf1[idx], dtype=torch.float32),
            "nt_encodings_rf2": torch.as_tensor(self.nt_encodings_rf2[idx], dtype=torch.float32),
            "aa_encodings_rf2": {key: val[idx] for key, val in self.aa_encodings_rf2.items()},
            "labels_rf2": torch.from_numpy(self.labels_rf2[idx]) if isinstance(self.labels_rf2[idx], np.ndarray) else torch.tensor(self.labels_rf2[idx], dtype=torch.float32),
            "label_encodings": torch.from_numpy(self.label_encodings[idx]) if isinstance(self.label_encodings[idx], np.ndarray) else torch.tensor(self.label_encodings[idx], dtype=torch.float32),
            "seq_desc": self.seq_desc[idx],
        }
        return item

    def __len__(self):
        return len(self.label_encodings)


def encode_data(processed_samples_df, max_len, tokenizer=None, max_aa_len=max_aa_len):
    """
    Encode data samples to fit model input format.

    Args:
        processed_samples_df (dataframe): Dataframe with input dataset.
        max_len (int): Max ESM model input length
        tokenizer (AutoTokenizer): Specific ESM tokenizer
        max_aa_len (int): maximum amino acid input length; max_len without special tokens (CLS and EOS)

    Returns:
        - dataset (dict): nested dictionary with data formatted to fit model input.
        - list of sequence types.
    """

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/esm2_t6_8M_UR50D",
            do_lower_case=False,
        )

    # Initialize
    encodings_nt = {}
    encodings_aa = {}
    labels = {}
    max_nt_len = max_aa_len * 3

    # Label processing; shared label sequence (mapped from rf0, rf1, rf2 labels)
    if isinstance(processed_samples_df["label_encodings"].iloc[0], str):
        processed_samples_df["label_encodings"] = processed_samples_df["label_encodings"].apply(eval)

    # Convert overall (shared) labels to arrays and pad to max length
    label_arrays = [np.array(x, dtype=np.int8) for x in processed_samples_df["label_encodings"]]
    pad_positions = max_len - 2
    padded_labels = np.full((len(label_arrays), pad_positions), -1, dtype=np.int8)

    for i, arr in enumerate(label_arrays):
        length = min(len(arr), pad_positions)
        padded_labels[i, :length] = arr[:length]

    # Rest of data is processed separately for each RF
    for rf in ["rf0", "rf1", "rf2"]:
        # ====Label processing====#
        if isinstance(processed_samples_df[f"{rf}_labels"].iloc[0], str):
            processed_samples_df[f"{rf}_labels"] = processed_samples_df[f"{rf}_labels"].apply(eval)

        # Convert to numpy arrays more efficiently
        label_arrays = [np.array(x, dtype=np.int8) for x in processed_samples_df[f"{rf}_labels"]]
        labels[rf] = label_arrays

        # ====Nucleotide sequence processing====#
        # Pad the sequences
        processed_samples_df[f"{rf}_seq_nt"] = processed_samples_df[f"{rf}_seq_nt"].apply(lambda seq: seq + "N" * (max_nt_len - len(seq)) if len(seq) < max_nt_len else seq)

        nt_sequences = [one_hot_encode(seq) for seq in processed_samples_df[f"{rf}_seq_nt"]]  # [max_nt_len, 4, num_seqs]

        # Process nt_sequences to codon-based format
        nt_encodings_rf = process_nt_sequences_to_codons(nt_sequences, max_aa_len)
        encodings_nt[rf] = nt_encodings_rf

        # ====Amino acid sequence processing====#
        aa_sequences = processed_samples_df[f"{rf}_seq_aa"].tolist()

        # Tokenize with strict length control
        aa_encodings_rf = tokenizer(
            aa_sequences,
            padding="max_length",  # Pad all to max_length
            max_length=max_len,  # CLS + 100 AA + EOS = 102
            truncation=True,  # Cut longer sequences
            return_tensors="pt",  # Return PyTorch tensors
        )

        encodings_aa[rf] = aa_encodings_rf

    # Get sequence types
    seq_descriptions = processed_samples_df["seq_desc"].tolist()

    # Get processed dataset
    dataset = SeqDataset(
        encodings_nt["rf0"],
        encodings_aa["rf0"],
        labels["rf0"],
        encodings_nt["rf1"],
        encodings_aa["rf1"],
        labels["rf1"],
        encodings_nt["rf2"],
        encodings_aa["rf2"],
        labels["rf2"],
        padded_labels,
        seq_descriptions,
    )

    return dataset, list(set(seq_descriptions))


def load_and_process_data(max_len, dataset_size):
    """
    Main function that loads and processes all data efficiently.

    Args:
        max_len: The model input length (amino acid/codon length + 2 for special tokens)

    Returns:
        train_data: The encoded training dataset
        val_data: The subsampled and encoded validation dataset
        sequence_types (list): list of all sequence types in dataset
        seq_type_desc_fracs: dict with distribution of sequence types in training data set
    """
    # Load data
    train_set = pd.read_csv(
        f"{input_data_dir_path}/datasets_model/train_{dataset_size}.csv.gz",  # Hyperparameter tuning on the 100 genome dataset
        index_col=None,
        compression="gzip",
    )

    val_set = pd.read_csv(f"{input_data_dir_path}/datasets_model/val.csv.gz", index_col=None, compression="gzip")

    # Create a combined stratification label for accession and sequence type
    val_set["accession_seq_desc_merged"] = val_set["accession"].astype(str) + "_" + val_set["seq_desc"].astype(str)  # modify

    ##Validate on 115k samples = 5 % of val set (0.05) following the original distribution stratified on accession and sequence type
    val_set = val_set.groupby("accession_seq_desc_merged", group_keys=False).apply(lambda x: x.sample(frac=frac_val, random_state=42))  # modify, 0.25

    # Create a combined stratification label for accession and sequence type
    train_set["accession_seq_desc_merged"] = train_set["accession"].astype(str) + "_" + train_set["seq_desc"].astype(str)  # DELETE

    ##Validate on 115k samples = 5 % of val set (0.05) following the original distribution stratified on accession and sequence type
    train_set = train_set.groupby("accession_seq_desc_merged", group_keys=False).apply(lambda x: x.sample(frac=frac_train, random_state=42))  ##DELETE

    seq_type_desc_fracs = (val_set["seq_desc"].value_counts(normalize=True)).to_dict()  # MODIFY
    seq_type_desc_fracs_train = (train_set["seq_desc"].value_counts(normalize=True)).to_dict()  # MODIFY

    print("Training data samples : ", train_set.shape[0], flush=True)
    print("Validation data samples during training: ", val_set.shape[0], flush=True)
    print("Distribution of sequence types in training set:", seq_type_desc_fracs_train, flush=True)
    print("Distribution of sequence types in validation set:", seq_type_desc_fracs, flush=True)

    # Create tokenizer once and reuse
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", do_lower_case=False)

    # Process training data
    train_data, sequence_types = encode_data(train_set, max_len, tokenizer)

    # Free memory after encoding
    del train_set
    gc.collect()

    # Process validation data
    val_data, _ = encode_data(val_set, max_len, tokenizer)

    # Free memory after encoding
    del val_set, tokenizer
    gc.collect()

    return train_data, val_data, sequence_types, seq_type_desc_fracs


# Define model architecture
class SequenceEncoder(nn.Module):
    """
    Sequence encoder using the pretrained ESM-2.

    Args:
        esm2_model (str): Name or path of the pretrained ESM-2 model to load
        dropout_rate_1 (float): Dropout rate for regularization layer applied after ESM-2
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing to reduce memory usage

    Attributes:
        pretrained_model_aa (AutoModel): Pretrained ESM-2 model with all layers initially frozen
        dropout_1 (nn.Dropout): Dropout layer for regularization before transformer head
    """

    def __init__(self, esm2_model, dropout_rate_1, use_gradient_checkpointing=False):
        super(SequenceEncoder, self).__init__()

        # Load pretrained ESM-2 model for amino acid sequences
        self.pretrained_model_aa = AutoModel.from_pretrained(esm2_model)

        # Enable gradient checkpointing to reduce memory usage (trades compute for memory)
        if use_gradient_checkpointing:
            self.pretrained_model_aa.gradient_checkpointing_enable() 
            #self.pretrained_model_aa.enable_input_require_grads() # Necessary for fine-tuning with gradient checkpointing to allow gradients to flow back into the model inputs
            print("Gradient checkpointing enabled for ESM-2 model", flush=True)

        self.num_layers = len(self.pretrained_model_aa.encoder.layer)

        # Freeze ALL layers of ESM-2 initially for staged training
        for param in self.pretrained_model_aa.parameters():
            param.requires_grad = False

        # Additional dropout layer for regularization after encoding sequences
        self.dropout_1 = nn.Dropout(dropout_rate_1)

        # Track which layers to unfreeze later
        self.unfreeze_start = None

    def prepare_for_unfreezing(self, unfreeze_fraction=0.5):
        """
        Prepare to unfreeze top layers later. This just calculates which layers
        will be unfrozen but doesn't actually unfreeze them yet.

        Args:
            unfreeze_fraction: Fraction of top layers to eventually unfreeze (default 0.5 = top half)
        """
        self.unfreeze_start = int(self.num_layers * (1 - unfreeze_fraction))
        print(f"Prepared to unfreeze layers {self.unfreeze_start} to {self.num_layers - 1} (top {unfreeze_fraction * 100}%)", flush=True)

    def unfreeze_top_layers(self, lr_esm2, optimizer, warmup_factor=0.1):
        """
        Unfreeze the top layers and add them to the optimizer with warmup.

        Args:
            lr_esm2: Learning rate for ESM-2 layers
            optimizer: The optimizer to add parameters to
            warmup_factor: Initial LR will be lr_esm2 * warmup_factor (default 0.1 = 10% of target)

        Returns:
            int: Index of the new parameter group in the optimizer
        """
        if self.unfreeze_start is None:
            raise ValueError("Must call prepare_for_unfreezing() before unfreeze_top_layers()")

        # Unfreeze the top layers
        unfrozen_count = 0
        for i, layer in enumerate(self.pretrained_model_aa.encoder.layer):
            if i >= self.unfreeze_start:
                for param in layer.parameters():
                    param.requires_grad = True
                    unfrozen_count += 1

        print(f"Unfrozen {unfrozen_count} parameters in layers {self.unfreeze_start} to {self.num_layers - 1}", flush=True)

        # Collect parameters from unfrozen layers
        unfrozen_params = [p for layer in self.pretrained_model_aa.encoder.layer[self.unfreeze_start :] for p in layer.parameters()]

        # Add to optimizer with warmup (start at reduced LR)
        initial_lr = lr_esm2 * warmup_factor
        optimizer.add_param_group({"params": unfrozen_params, "lr": initial_lr})

        param_group_idx = len(optimizer.param_groups) - 1
        print(f"Added {len(unfrozen_params)} parameters to optimizer with initial LR={initial_lr:.2e} (warmup factor={warmup_factor})", flush=True)

        return param_group_idx

    def warmup_lr(self, optimizer, param_group_idx, target_lr, current_step, warmup_steps):
        """
        Gradually increase learning rate for unfrozen layers during warmup period.

        Args:
            optimizer: The optimizer
            param_group_idx: Index of the parameter group to adjust
            target_lr: Target learning rate to reach
            current_step: Current step in warmup (0 to warmup_steps)
            warmup_steps: Total warmup steps
        """
        if current_step >= warmup_steps:
            new_lr = target_lr
        else:
            # Linear warmup
            warmup_factor = current_step / warmup_steps
            new_lr = target_lr * warmup_factor

        optimizer.param_groups[param_group_idx]["lr"] = new_lr

    def forward(self, x_aa, attention_mask_aa):
        """
        Forward pass through the sequence encoder.

        Args:
            x_aa (torch.Tensor): Tokenized amino acid sequences of shape (batch_size, seq_len)
            attention_mask_aa (torch.Tensor): Attention mask of shape (batch_size, seq_len)

        Returns:
            embeddings_aa (torch.Tensor): Amino acid embeddings with CLS/EOS removed, shape (batch_size, seq_len-2, hidden_size)
            attention_mask_trimmed (torch.Tensor): Attention mask with CLS/EOS removed, shape (batch_size, seq_len-2)
        """
        # Extract features from pretrained ESM-2 model
        features_aa = self.pretrained_model_aa(x_aa, attention_mask=attention_mask_aa)

        # Get last hidden state: [batch_size, tokens, hidden_size]
        sequence_output_aa = features_aa["last_hidden_state"]

        # Remove CLS and EOS token embeddings: [batch_size, aa_seq_len, hidden_size]
        sequence_output_aa = sequence_output_aa[:, 1:-1, :]

        # Apply dropout before transformer head
        embeddings_aa = self.dropout_1(sequence_output_aa)

        # Remove CLS/EOS from attention mask for transformer head
        attention_mask_trimmed = attention_mask_aa[:, 1:-1]

        return embeddings_aa, attention_mask_trimmed


class TransformerEncoderBlock(nn.Module):
    """
    Neural network module that applies a Transformer encoder block to encoded sequences and adds a linear layer to fit CRF input.

    Args:
        hidden_size (int): The dimensionality of the input and output features for the Transformer encoder.
        num_layers (int): Number of Transformer encoder layers to stack.
        n_attention_heads (int): Number of attention heads in each Transformer encoder layer.
        dropout_rate_encoder (float): Dropout rate applied after normalization and within the encoder layers.
        act_function (str or Callable): Activation function to use in the feedforward network of the encoder layers.
        num_labels (int): Number of output classes

    Attributes:
        encoder (nn.TransformerEncoder): Stacked Transformer encoder layers.
        classifier (nn.Linear): Linear layer mapping the encoder output to class logits.
        norm (nn.LayerNorm): Layer normalization applied after the encoder.
        dropout (nn.Dropout): Dropout layer applied after normalization.
        layers (int): Number of encoder layers.
    """

    def __init__(self, hidden_size, num_layers, n_attention_heads, dropout_rate_encoder, act_function, num_labels):
        super().__init__()

        hidden_size_merged = hidden_size + 12  # 12 for codon one-hot encoded; hidden_size for amino acid representation from ESM2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size_merged,
            nhead=n_attention_heads,
            dim_feedforward=4 * hidden_size_merged,
            dropout=dropout_rate_encoder,
            activation=act_function,
        )

        self.layers = num_layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size_merged, num_labels)
        self.norm = nn.LayerNorm(hidden_size_merged)

    def forward(self, encoded_seqs_nt, encoded_embeddings_aa, trimmed_attention_mask):
        """
        Forward pass through the Transformer encoder block and RF-specific linear classifier.

        Args:
            encoded_seqs_nt (torch.Tensor): Nucleotide encodings with shape (batch_size, seq_len, 12) (one-hot encoded codons).
            encoded_embeddings_aa (torch.Tensor): Amino acid embeddings with shape (batch_size, seq_len, hidden_size) from pretrained ESM-2
            trimmed_attention_mask (torch.Tensor): Boolean mask of shape (batch_size, seq_len) where 1s are valid tokens and 0s are padding positions.

        Returns:
            torch.Tensor:
                Logits of shape (batch_size, seq_len, num_labels) representing class scores for each token position.
        """

        # Concatenate ESM-2 embeddings and one hot encoded codons
        combined_codon_and_aa_embeddings = torch.cat([encoded_embeddings_aa, encoded_seqs_nt], dim=-1)

        combined_codon_and_aa_embeddings = combined_codon_and_aa_embeddings.permute(1, 0, 2)  # [seq_len, batch, hidden + 12]
        attention_mask_transformer = ~trimmed_attention_mask.bool()

        # Pass through transformer encoder layers
        combined_codon_and_aa_embeddings = self.encoder(combined_codon_and_aa_embeddings, src_key_padding_mask=attention_mask_transformer)

        combined_codon_and_aa_embeddings = combined_codon_and_aa_embeddings.permute(1, 0, 2)  # [batch, seq_len, hidden + 12]

        # Apply layer normalization
        combined_codon_and_aa_embeddings = self.norm(combined_codon_and_aa_embeddings)

        # Get RF-specific class logits out
        logits = self.linear(combined_codon_and_aa_embeddings)  # [batch, seq_len, num_labels]

        return logits


class LinearChainCRF(nn.Module):
    """
    Neural network module that applies a Conditional Random Field (CRF) layer for structured prediction.
    Designed to handle dynamic reading-frame (RF) combination labels and enforce biologically
    constrained transitions between RF states.

    Args:
        mapping_dict_to_class (dict): Mapping from integer label indices to tuples representing the corresponding reading-frame combination (e.g., `{0: (0, 1, 2), 1: (1, 3, 5), ...}`).
        num_encoded_labels (int, optional): Total number of class labels. If not provided, it is inferred from `mapping_dict_to_class`.

    Attributes:
        shared_rf_labels_mapping (dict): Stores the mapping from label indices to RF combinations.
        crf (torchcrf.CRF): Linear-chain Conditional Random Field layer that models label dependencies across the sequence.
        legal_transitions (dict): Defines valid transitions between RF states for each of the three reading frames.
        biologically_valid_mask (torch.BoolTensor): Mask indicating which transitions are allowed (True) or constrained (False).
        frequent_transition_mask (torch.BoolTensor): Mask indicating which transitions are frequent self-transitions.
    """

    def __init__(self, mapping_dict_to_class, transition_weight, num_encoded_labels=None, label_classes=label_classes):
        super().__init__()

        # Load shared class mapping
        self.shared_rf_labels_mapping = mapping_dict_to_class

        # Determine number of classes if not provided
        if num_encoded_labels is None:
            num_encoded_labels = len(self.shared_rf_labels_mapping)

        self.crf = CRF(num_tags=num_encoded_labels, batch_first=True)

        # In LinearChainCRF.__init__, add label_classes parameter and use it:
        if label_classes == 6:
            self.legal_transitions = {
                0: {0, 2, 4}, 1: {1, 3, 5}, 2: {1, 5}, 
                3: {0, 2, 4}, 4: {1, 3, 5}, 5: {0, 2, 4}
            }
        elif label_classes == 4:  # 4 classes
            self.legal_transitions = {
                0: {0, 2},
                1: {1, 3},
                2: {1},
                3: {0, 2}
            }
        else:
            raise ValueError("label_classes must be either 4 or 6.")

        self.biologically_valid_mask = torch.ones_like(self.crf.transitions, dtype=torch.bool)
        self.frequent_transition_mask = torch.zeros_like(self.crf.transitions, dtype=torch.bool)

        self._create_biologically_valid_mask()
        self._create_frequent_transition_mask()

        # Initialize transitions with three-tier scheme + small noise for diversity
        with torch.no_grad():
            # Stage 1: All illegal transitions → -10 (forbidden)
            self.crf.transitions[~self.biologically_valid_mask] = -10

            # Stage 2: Legal but infrequent transitions → transition_weight (penalized)
            legal_infrequent = self.biologically_valid_mask & ~self.frequent_transition_mask
            self.crf.transitions[legal_infrequent] = transition_weight

            # Stage 3: Frequent self-transitions → 0 (neutral/encouraged)
            self.crf.transitions[self.frequent_transition_mask] = 0

    def _is_legal_transition(self, from_rf, to_rf):
        """Check if transition from one RF combination to another is legal"""
        from_rf0, from_rf1, from_rf2 = from_rf
        to_rf0, to_rf1, to_rf2 = to_rf

        # All three RFs must have legal transitions
        rf0_legal = to_rf0 in self.legal_transitions[from_rf0]
        rf1_legal = to_rf1 in self.legal_transitions[from_rf1]
        rf2_legal = to_rf2 in self.legal_transitions[from_rf2]

        return rf0_legal and rf1_legal and rf2_legal

    def _create_biologically_valid_mask(self):
        """
        Create a mask indicating which transitions should be constrained.
        False = constrained (will be set to penalty), True = learnable
        """
        num_labels = len(self.shared_rf_labels_mapping)
        legal_count = 0
        illegal_count = 0

        # Check all possible transitions between the labels
        for from_label in range(num_labels):
            for to_label in range(num_labels):
                from_rf = self.shared_rf_labels_mapping[from_label]
                to_rf = self.shared_rf_labels_mapping[to_label]

                if self._is_legal_transition(from_rf, to_rf):
                    # Legal transition - keep as learnable (True)
                    self.biologically_valid_mask[from_label, to_label] = True
                    legal_count += 1
                else:
                    # Illegal transition - mark as constrained (False)
                    self.biologically_valid_mask[from_label, to_label] = False
                    illegal_count += 1

        print(f"Legal transitions: {legal_count}", flush=True)
        print(f"Illegal transitions: {illegal_count}", flush=True)
        print(f"Total transitions: {legal_count + illegal_count}", flush=True)
        print(f"Percentage legal: {legal_count / (legal_count + illegal_count) * 100:.1f}%", flush=True)

    def _create_frequent_transition_mask(self):
        """
        Create a mask for frequent self-transitions.
        These are transitions from a label to itself for the special RF combinations:
        (0,0,0)->(0,0,0), (1,0,0)->(1,0,0), (0,1,0)->(0,1,0), (0,0,1)->(0,0,1)
        """
        # Define the frequent RF combinations
        frequent_rf_combinations = {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)}

        # Find label indices corresponding to these RF combinations
        frequent_labels = []
        for label_idx, rf_combo in self.shared_rf_labels_mapping.items():
            if rf_combo in frequent_rf_combinations:
                frequent_labels.append(label_idx)

        # Mark self-transitions for these labels
        frequent_count = 0
        for label in frequent_labels:
            self.frequent_transition_mask[label, label] = True
            frequent_count += 1

        print(f"Frequent self-transitions: {frequent_count}", flush=True)
        print(f"Frequent transition labels: {frequent_labels}", flush=True)

    def forward(self, logits, attention_mask, labels=None): #MODIFY
        """
        Forward pass with CRF layer.

        Args:
            logits (torch.Tensor): Emission scores of shape (batch_size, seq_len, num_labels).
            attention_mask (torch.Tensor): Mask where 1 indicates valid tokens.
            labels (torch.Tensor, optional): Gold label indices for training.
            sequence_class (torch.Tensor, optional): Sequence-level class indices for weighted loss.

        Returns:
            dict:
            During training (`labels` provided):
                    - **'loss' (torch.Tensor)**: Mean CRF loss over the batch.
                    - **'logits' (torch.Tensor)**: Input logits passed through the CRF.
                During inference (`labels` omitted):
                    - **'predictions' (list[list[int]])**: Decoded most probable label sequence per sample.
                    - **'logits' (torch.Tensor)**: Input logits passed through the CRF.
        """
        # Training
        if labels is not None:

            # Use label-based mask instead of attention mask
            crf_mask = (labels != -1)
            
            # Replace -1 with 0 in labels (masked positions don't matter, but -1 is invalid index)
            safe_labels = labels.clone()
            safe_labels[safe_labels == -1] = 0
            
            log_likelihood = self.crf(logits, safe_labels, mask=crf_mask, reduction="none")
            loss = -log_likelihood.mean()

            return {"loss": loss, "logits": logits}
        
        else:
            crf_mask = attention_mask.bool()
            predictions = self.crf.decode(logits, mask=crf_mask)
            return {"predictions": predictions, "logits": logits}

class CDSPredictor(nn.Module):
    """
    Full model for CDS prediction combining:
      - Pretrained ESM-2 amino acid embeddings,
      - Transformer encoders per reading frame (RF0–RF2),
      - And a CRF layer for structured, frame-consistent predictions.

    Args:
        esm2_model (nn.Module): Pretrained ESM-2 model used to extract amino acid embeddings.
        num_layers (int): Number of Transformer encoder layers per reading frame.
        n_attention_heads (int): Number of attention heads in each Transformer layer.
        dropout_rate_1 (float): Dropout rate applied in the sequence encoder.
        dropout_rate_2 (float): Dropout rate applied within the Transformer encoder layers.
        act_function (str or Callable): Activation function used in Transformer feedforward layers.
        num_encoded_labels (int): Number of combined label states used by the CRF.
        encoded_labels_mapping (dict): Mapping from integer label indices to RF combination tuples.

    Attributes:
        sequence_encoder (SequenceEncoder): Module that extracts amino acid embeddings from the pretrained ESM-2 model.
        TransformerEncoderBlock (TransformerEncoderBlock): Transformer encoder applied independently to each reading frame.
        linear_transform (nn.Linear): Linear projection layer mapping concatenated RF outputs (3*num_labels) to num_encoded_labels.
        CRF (LinearChainCRF): Conditional Random Field layer enforcing structured transitions between predicted RF combinations.
    """

    def __init__(
        self,
        esm2_model,
        num_layers,
        n_attention_heads,
        dropout_rate_1,
        dropout_rate_2,
        act_function,
        transition_weight,
        num_encoded_labels,
        encoded_labels_mapping,
        label_classes=label_classes,
        use_gradient_checkpointing=False,
    ):
        super(CDSPredictor, self).__init__()

        # Extract amino acid representations from pretrained ESM-2 model
        self.sequence_encoder = SequenceEncoder(esm2_model, dropout_rate_1, use_gradient_checkpointing)

        # Transformer encoder block applied separately to each reading frame
        self.TransformerEncoderBlock = TransformerEncoderBlock(
            hidden_size=self.sequence_encoder.pretrained_model_aa.config.hidden_size,
            num_layers=num_layers,
            n_attention_heads=n_attention_heads,
            dropout_rate_encoder=dropout_rate_2,
            act_function=act_function,
            num_labels=label_classes,
        )

        # Linear layer to combine outputs from the 3 reading frames (3 * 6 logits -> num_encoded_labels)
        self.linear_transform = nn.Linear(3 * label_classes, num_encoded_labels)
        # CRF layer for structured prediction with transition constraints
        self.CRF = LinearChainCRF(
            mapping_dict_to_class=encoded_labels_mapping,
            transition_weight=transition_weight,
            num_encoded_labels=num_encoded_labels,
            label_classes=label_classes,
        )

    def forward(
        self,
        encoded_seqs_nt_rf0,
        x_aa_rf0,
        attention_mask_aa_rf0,
        encoded_seqs_nt_rf1,
        x_aa_rf1,
        attention_mask_aa_rf1,
        encoded_seqs_nt_rf2,
        x_aa_rf2,
        attention_mask_aa_rf2,
        labels=None,
    ):
        """
        Forward pass through the full CDS prediction model.

        Args:
            encoded_seqs_nt_rf{0,1,2} (torch.Tensor): One-hot codon encodings for each reading frame, shape (batch_size, seq_len, 12).
            x_aa_rf{0,1,2} (torch.Tensor): Amino acid token embeddings for each RF, shape (batch_size, seq_len, hidden_size).
            attention_mask_aa_rf{0,1,2} (torch.Tensor): Boolean masks marking valid tokens, shape (batch_size, seq_len).
            labels (torch.Tensor, optional): Ground-truth encoded, shared labels for CRF training.
            sequence_class (torch.Tensor, optional): Sequence-level class indices for weighted CRF loss.

        Returns:
            dict:
                If training → {'loss': torch.Tensor, 'logits': torch.Tensor}
                If inference → {'predictions': list[list[int]], 'logits': torch.Tensor}

        """

        # Encode amino acid sequences for each reading frame
        encoded_embeddings_aa_rf0, trimmed_attention_mask_rf0 = self.sequence_encoder(x_aa_rf0, attention_mask_aa_rf0)
        encoded_embeddings_aa_rf1, trimmed_attention_mask_rf1 = self.sequence_encoder(x_aa_rf1, attention_mask_aa_rf1)
        encoded_embeddings_aa_rf2, trimmed_attention_mask_rf2 = self.sequence_encoder(x_aa_rf2, attention_mask_aa_rf2)

        # Process each RF through its transformer encoder blocks
        logits_rf0 = self.TransformerEncoderBlock(
            encoded_seqs_nt=encoded_seqs_nt_rf0,
            encoded_embeddings_aa=encoded_embeddings_aa_rf0,
            trimmed_attention_mask=trimmed_attention_mask_rf0,
        )
        logits_rf1 = self.TransformerEncoderBlock(
            encoded_seqs_nt=encoded_seqs_nt_rf1,
            encoded_embeddings_aa=encoded_embeddings_aa_rf1,
            trimmed_attention_mask=trimmed_attention_mask_rf1,
        )
        logits_rf2 = self.TransformerEncoderBlock(
            encoded_seqs_nt=encoded_seqs_nt_rf2,
            encoded_embeddings_aa=encoded_embeddings_aa_rf2,
            trimmed_attention_mask=trimmed_attention_mask_rf2,
        )  # output: [codon_seq_len, C]

        # Concatenate logits from all reading frames along the feature (class logit) dimension
        combined_codon_and_aa_embeddings = torch.cat([logits_rf0, logits_rf1, logits_rf2], dim=-1)  # output: [codon_seq_len, 3*C]

        # Map combined frame representations to encoded, shared label space
        logits_encoded_labels = self.linear_transform(combined_codon_and_aa_embeddings)  # output: [codon_seq_len, num_encoded_labels]

        # Apply CRF for structured decoding or training (same attention mask applies to all RFs)
        output = self.CRF(
            logits=logits_encoded_labels,
            attention_mask=trimmed_attention_mask_rf0,  # Input any trimmed attention mask; applies to same positions as before
            labels=labels,
        )

        return output


def print_model_dimensions(model):
    """Print model dimensions if needed."""
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}", flush=True)


def count_parameters(model):
    """Print number of model parameters; both learnable and total"""
    total_params_learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total parameters: {total_params:,}", flush=True)
    print(f"Total trainable parameters: {total_params_learnable:,}", flush=True)


def initialize_model(device, num_layers, n_attention_heads, dropout_rate_1, dropout_rate_2, act_function, transition_weight, label_classes, use_gradient_checkpointing=False):
    """
    Initialize the model and move it to the specified device.

    Args:
        device_type (str): The device to use for computation ("cuda", "mps", or "cpu").
        num_layers (int): Number of Transformer encoder layers per reading frame.
        n_attention_heads (int): Number of attention heads in each Transformer encoder layer.
        dropout_rate_1 (float): Dropout rate applied in the sequence encoder.
        dropout_rate_2 (float): Dropout rate applied within the Transformer encoder layers.
        act_function (str or Callable): Activation function used in Transformer feedforward layers.
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing to reduce memory.

    Returns:
        model (nn.Module): The initialized model.
        mapping_dict_to_class (dict): dictionary mapping the shared label encodings to 3D rf-specific label combinations
    """

    print("Running on: ", device, flush=True)

    with open(f"{input_data_dir_path}/label_mappings/mapping_to_3d_vector.pkl", "rb") as mapping_file:
        mapping_dict_to_class = pickle.load(mapping_file)

    num_encoded_labels = len(mapping_dict_to_class.keys())
    print(f"Number of encoded label classes: {num_encoded_labels}", flush=True)

    model = CDSPredictor(
        esm2_model=esm2_model,
        num_layers=num_layers,
        n_attention_heads=n_attention_heads,
        dropout_rate_1=dropout_rate_1,
        dropout_rate_2=dropout_rate_2,
        act_function=act_function,
        transition_weight=transition_weight,
        num_encoded_labels=num_encoded_labels,
        encoded_labels_mapping=mapping_dict_to_class,
        label_classes=label_classes,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    model.to(device)

    if device.type == "cuda":
        print(f"Memory Allocated after loading model: {torch.cuda.memory_allocated(device) / 1024**3} GB", flush=True)

    count_parameters(model)
    # print_model_dimensions(model)

    return model, mapping_dict_to_class


def calculate_sequence_accuracy_metrics(true_labels_list, predictions_list, sequence_types_list=None):
    """
    Calculate sequence-level accuracy metrics including:
        - MCC for overall and for each sequence type specifically.
        - Sequence type accuracy metrics

    Args:
        true_labels_list: List of numpy arrays, each containing true labels for one sequence
        predictions_list: List of numpy arrays, each containing predictions for one sequence
        sequence_types_list: Optional list of sequence types/descriptions corresponding to each sequence

    Returns:
        results (dict): Dictionary containing accuracy metrics including overall MCC and type-specific metrics
    """

    if len(true_labels_list) != len(predictions_list):
        raise ValueError("Mismatch in number of sequences")

    if sequence_types_list is not None and len(sequence_types_list) != len(true_labels_list):
        raise ValueError("Mismatch in number of sequence types")

    total_sequences = len(true_labels_list)

    # Initalize
    perfect_sequences = 0
    high_accuracy_sequences = 0
    sequence_accuracies = []
    batch_size = 1000  # Process sequences in batches to save memory
    all_true_labels = []
    all_predictions = []

    # For sequence type-specific metrics
    type_specific_data = {}
    if sequence_types_list is not None:
        unique_types = list(set(sequence_types_list))
        for seq_type in unique_types:
            type_specific_data[seq_type] = {
                "true_labels": [],
                "predictions": [],
                "sequence_accuracies": [],
                "perfect_sequences": 0,
                "high_accuracy_sequences": 0,
                "total_sequences": 0,
            }

    for i in range(0, total_sequences, batch_size):
        batch_true = true_labels_list[i : i + batch_size]
        batch_pred = predictions_list[i : i + batch_size]
        batch_types = sequence_types_list[i : i + batch_size] if sequence_types_list else None

        batch_accuracies = []
        batch_all_true = []
        batch_all_pred = []

        for idx, (true_labels, predictions) in enumerate(zip(batch_true, batch_pred)):
            true_labels = np.asarray(true_labels)
            predictions = np.asarray(predictions)

            # Add batch to overall arrays
            batch_all_true.extend(true_labels.tolist())
            batch_all_pred.extend(predictions.tolist())

            # Calculate accuracy
            accuracy = (true_labels == predictions).mean()
            batch_accuracies.append(accuracy)

            if accuracy == 1.0:
                perfect_sequences += 1
            if accuracy > 0.9:
                high_accuracy_sequences += 1

            # Handle type-specific data
            if batch_types is not None:
                seq_type = batch_types[idx]
                type_data = type_specific_data[seq_type]

                # Add to type-specific arrays
                type_data["true_labels"].extend(true_labels.tolist())
                type_data["predictions"].extend(predictions.tolist())
                type_data["sequence_accuracies"].append(accuracy)
                type_data["total_sequences"] += 1

                if accuracy == 1.0:
                    type_data["perfect_sequences"] += 1
                if accuracy > 0.9:
                    type_data["high_accuracy_sequences"] += 1

        sequence_accuracies.extend(batch_accuracies)
        all_true_labels.extend(batch_all_true)
        all_predictions.extend(batch_all_pred)

        # Clear batch data
        del batch_true, batch_pred, batch_accuracies, batch_all_true, batch_all_pred
        if batch_types is not None:
            del batch_types

    # Convert to numpy arrays
    all_true_labels = np.array(all_true_labels)
    all_predictions = np.array(all_predictions)

    # alculate overall MCC efficiently
    try:
        overall_mcc = matthews_corrcoef(all_true_labels, all_predictions)  # Multi-class MCC
        if np.isnan(overall_mcc):
            overall_mcc = 0.0
    except:
        overall_mcc = 0.0

    # Build results
    results = {
        "fraction_perfect_sequences": perfect_sequences / total_sequences,
        "fraction_high_accuracy_sequences": high_accuracy_sequences / total_sequences,
        "overall_mcc": overall_mcc,
        "accuracy": np.mean(sequence_accuracies),
    }

    # Calculate type-specific metrics
    if sequence_types_list is not None:
        type_metrics = {}
        for seq_type, type_data in type_specific_data.items():
            if type_data["total_sequences"] > 0:
                # Convert type-specific data to numpy arrays
                type_true = np.array(type_data["true_labels"])
                type_pred = np.array(type_data["predictions"])

                # Calculate type-specific MCC
                try:
                    type_mcc = matthews_corrcoef(type_true, type_pred)  # Multi-class MCC
                    if np.isnan(type_mcc):
                        type_mcc = 0.0
                except:
                    type_mcc = 0.0

                # Calculate type-specific metrics
                type_metrics[f"{seq_type}_mcc"] = type_mcc
                type_metrics[f"{seq_type}_accuracy"] = np.mean(type_data["sequence_accuracies"])
                type_metrics[f"{seq_type}_fraction_perfect"] = type_data["perfect_sequences"] / type_data["total_sequences"]
                type_metrics[f"{seq_type}_fraction_high_accuracy"] = type_data["high_accuracy_sequences"] / type_data["total_sequences"]

        # Add type-specific metrics to results
        results.update(type_metrics)

    return results


class CategoricalLossTracker: #MODIFY
    """
    A class to track losses for different sequence types during training.
    Properly weights by number of sequences per category.
    """

    def __init__(self, categories):
        self.categories = categories
        self.total_loss = {cat: 0.0 for cat in categories}
        self.total_count = {cat: 0 for cat in categories}

    def update(self, category, loss, count):
        # Accumulate total loss and count for proper averaging
        self.total_loss[category] += loss * count
        self.total_count[category] += count

    def get_metrics(self):
        # Calculate the properly weighted mean loss for each category
        metrics = {}
        for cat in self.categories:
            if self.total_count[cat] > 0:
                metrics[cat] = self.total_loss[cat] / self.total_count[cat]
            else:
                metrics[cat] = 0.0
        return metrics


def log_evaluation_metrics(epoch, train_avg_loss, val_avg_loss, best_val_loss, tracker, sequence_metrics, val_times_counter, sequence_types):
    """
    Print evaluation metrics for the current epoch and prepare wandb logging dict.

    Args:
        epoch (int): Current training epoch.
        train_avg_loss (float): Average training loss for this epoch.
        val_avg_loss (float): Average validation loss for this epoch.
        best_val_loss (float): Best validation loss seen so far.
        tracker (object): Object providing access to per-sequence-type validation loss metrics via tracker.get_metrics().
        sequence_metrics (dict): Dictionary containing overall and type-specific sequence metrics, such as MCC, accuracy, fraction of perfect/high-accuracy sequences.
        val_times_counter (int): Counter tracking how many validation runs have been performed.
        sequence_types (list[str]): List of sequence types (e.g., categories) to include in type-specific metrics.

    Returns:
        tuple: (val_times_counter + 1, wandb_log dict) - Updated counter and metrics dict for logging.
    """

    # Build the print statement with type-specific metrics
    print_parts = [
        f"---Evaluation {val_times_counter}---\n",
        f"Train Loss: {train_avg_loss:.4f}\t\t",
        f"Val Loss: {val_avg_loss:.4f}\t\t",
    ]

    # Add overall metrics
    print_parts.extend(
        [
            f"Overall MCC: {sequence_metrics.get('overall_mcc', 0):.4f}\t\t",
            f"Overall Accuracy: {sequence_metrics.get('accuracy', 0):.4f}\n",
        ]
    )

    # Add loss metrics for each sequence type
    for seq_type in sequence_types:
        loss_val = tracker.get_metrics().get(seq_type, 0) 
        print_parts.append(f"Val Loss {seq_type}: {loss_val:.4f}\t\t")

    print_parts.append("\n")

    # Add type-specific MCC and accuracy metrics
    for seq_type in sequence_types:
        type_mcc = sequence_metrics.get(f"{seq_type}_mcc", 0)
        type_acc = sequence_metrics.get(f"{seq_type}_accuracy", 0)
        if type_mcc != 0 or type_acc != 0:  # Only show if we have data for this type
            print_parts.extend([f"MCC {seq_type}: {type_mcc:.4f}\t\t", f"Acc {seq_type}: {type_acc:.4f}\t\t"])

    # Print all metrics
    print("".join(print_parts), flush=True)

    # Build wandb logging dictionary
    wandb_log = {
        "epoch": epoch + 1,
        "train_loss": train_avg_loss,
        "val_loss": val_avg_loss,
        # Overall sequence metrics
        "val_fraction_perfect_sequences": sequence_metrics.get("fraction_perfect_sequences", 0),
        "val_fraction_high_accuracy_sequences": sequence_metrics.get("fraction_high_accuracy_sequences", 0),
        "val_overall_mcc": sequence_metrics.get("overall_mcc", 0),
        "val_accuracy": sequence_metrics.get("accuracy", 0),
    }

    # Add loss metrics for each sequence type
    for seq_type in sequence_types:
        wandb_log[f"val_loss_{seq_type}"] = tracker.get_metrics().get(seq_type, 0)

    # Add type-specific MCC and accuracy metrics
    for seq_type in sequence_types:
        # MCC metrics
        type_mcc = sequence_metrics.get(f"{seq_type}_mcc", 0)
        if type_mcc != 0:  # Only log if we have data
            wandb_log[f"val_mcc_{seq_type}"] = type_mcc

        # Accuracy metrics
        type_acc = sequence_metrics.get(f"{seq_type}_accuracy", 0)
        if type_acc != 0:  # Only log if we have data
            wandb_log[f"val_accuracy_{seq_type}"] = type_acc

        # Perfect and high accuracy sequence fractions
        type_perfect = sequence_metrics.get(f"{seq_type}_fraction_perfect", 0)
        if type_perfect != 0:
            wandb_log[f"val_fraction_perfect_{seq_type}"] = type_perfect

        type_high_acc = sequence_metrics.get(f"{seq_type}_fraction_high_accuracy", 0)
        if type_high_acc != 0:
            wandb_log[f"val_fraction_high_accuracy_{seq_type}"] = type_high_acc

    # Log lowest validation loss obtained throughout entire training
    wandb_log[f"best_val_loss"] = best_val_loss

    # Return the dict instead of logging - caller will consolidate all metrics
    return val_times_counter + 1, wandb_log


def training_iteration(i, batch, scaler, model, optimizer, device, train_losses, warmup_state=None):
    """
    Perform a single training iteration on one batch, including forward and backward passes,
    optional mixed precision, gradient clipping, and loss accumulation.

    Args:
        i (int): Index of the current batch in the epoch (used for logging/printing).
        batch (dict): Dictionary containing batch data with keys:
            - "nt_encodings_rf0/1/2": nucleotide encodings per reading frame.
            - "aa_encodings_rf0/1/2": amino acid token encodings per reading frame, each with keys "input_ids" and "attention_mask".
            - "label_encodings": target labels for the CRF layer (with -1 for padding positions).
            - "seq_desc": List of string sequence types (e.g., "coding", "non-coding", "transition_start").
        scaler (torch.cuda.amp.GradScaler or None): Mixed precision scaler; if None, regular precision is used.
        model (nn.Module): CDS prediction model with transformer and CRF layers.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        device (torch.device): Device for computation ("cuda", "cpu", etc.).
        train_losses (list): List storing training loss values per batch for the current epoch.
        seq_type_to_idx (dict): Mapping from sequence type strings to integer indices for weighted loss.
        warmup_state (dict, optional): Dictionary containing warmup state with keys:
            - 'counter': current warmup step
            - 'total_steps': total warmup steps
            - 'param_group_idx': optimizer parameter group index for ESM-2
            - 'target_lr': target learning rate for ESM-2

    Returns:
        tuple: (updated train_losses list, updated warmup_state dict or None)
    """

    # Move data to device
    inputs_nt_rf0 = batch["nt_encodings_rf0"].to(device, non_blocking=True)
    inputs_aa_rf0 = batch["aa_encodings_rf0"]["input_ids"].to(device, non_blocking=True)
    attention_mask_aa_rf0 = batch["aa_encodings_rf0"]["attention_mask"].to(device, non_blocking=True)

    inputs_nt_rf1 = batch["nt_encodings_rf1"].to(device, non_blocking=True)
    inputs_aa_rf1 = batch["aa_encodings_rf1"]["input_ids"].to(device, non_blocking=True)
    attention_mask_aa_rf1 = batch["aa_encodings_rf1"]["attention_mask"].to(device, non_blocking=True)

    inputs_nt_rf2 = batch["nt_encodings_rf2"].to(device, non_blocking=True)
    inputs_aa_rf2 = batch["aa_encodings_rf2"]["input_ids"].to(device, non_blocking=True)
    attention_mask_aa_rf2 = batch["aa_encodings_rf2"]["attention_mask"].to(device, non_blocking=True)

    encoded_labels = batch["label_encodings"].to(device, non_blocking=True).long()

    # DEBUG: log memory for first 5 training batches
    _debug_mem = torch.cuda.is_available() and i < 5
    if _debug_mem:
        print(f"[TRAIN MEM] batch {i} pre-forward  | alloc: {torch.cuda.memory_allocated(device)/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB", flush=True)

    # Forward pass; use mixed precision if available
    if scaler is not None:
        with autocast("cuda"):
            outputs = model(
                inputs_nt_rf0,
                inputs_aa_rf0,
                attention_mask_aa_rf0,
                inputs_nt_rf1,
                inputs_aa_rf1,
                attention_mask_aa_rf1,
                inputs_nt_rf2,
                inputs_aa_rf2,
                attention_mask_aa_rf2,
                labels=encoded_labels,
            )
            loss = outputs["loss"]
    else:
        # Regular forward pass without mixed precision
        outputs = model(
            inputs_nt_rf0,
            inputs_aa_rf0,
            attention_mask_aa_rf0,
            inputs_nt_rf1,
            inputs_aa_rf1,
            attention_mask_aa_rf1,
            inputs_nt_rf2,
            inputs_aa_rf2,
            attention_mask_aa_rf2,
            labels=encoded_labels,
        )
        loss = outputs["loss"]

    if _debug_mem:
        print(f"[TRAIN MEM] batch {i} post-forward | alloc: {torch.cuda.memory_allocated(device)/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB", flush=True)

    # Check for NaN loss before backward pass
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"WARNING: NaN/Inf loss detected at batch {i}! Skipping this batch.", flush=True)
        return train_losses, warmup_state

    # Backward pass
    optimizer.zero_grad()

    if scaler is not None:
        # Mixed precision backward pass
        scaler.scale(loss).backward()

        if _debug_mem:
            print(f"[TRAIN MEM] batch {i} post-backward | alloc: {torch.cuda.memory_allocated(device)/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB", flush=True)

        # Unscale before gradient clipping
        scaler.unscale_(optimizer)

        # Gradient clipping (CRITICAL for CRF stability!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
    else:
        # Regular backward pass
        loss.backward()

        if _debug_mem:
            print(f"[TRAIN MEM] batch {i} post-backward | alloc: {torch.cuda.memory_allocated(device)/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB", flush=True)

        # Gradient clipping (CRITICAL for CRF stability!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    if _debug_mem:
        print(f"[TRAIN MEM] batch {i} post-optim   | alloc: {torch.cuda.memory_allocated(device)/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB", flush=True)

    # Handle warmup if ESM-2 layers were unfrozen
    if warmup_state is not None and warmup_state["counter"] < warmup_state["total_steps"]:
        model.sequence_encoder.warmup_lr(
            optimizer,
            warmup_state["param_group_idx"],
            warmup_state["target_lr"],
            warmup_state["counter"],
            warmup_state["total_steps"],
        )
        warmup_state["counter"] += 1

    # Store loss value
    train_losses.append(loss.item())

    # Print transition matrix for checkpoint
    if i % 10000 == 0:
        print("CRF transition matrix sample (first 5x5):", flush=True)
        print(model.CRF.crf.transitions[:5, :5], flush=True)

    if i % 1000 == 0:
        # Cleanup every 1000th batch to prevent memory leaks
        del outputs, loss
        del inputs_nt_rf0, inputs_aa_rf0, attention_mask_aa_rf0
        del inputs_nt_rf1, inputs_aa_rf1, attention_mask_aa_rf1
        del inputs_nt_rf2, inputs_aa_rf2, attention_mask_aa_rf2
        del encoded_labels

    return train_losses, warmup_state


def show_examples(v_labels, padding_mask, logits, seq_descs_batch, mapping_dict_to_class, model, device, valid_mask):
    """
    Show hard examples as demonstration per validation loop

    Args:
        counter: batch counter
        v_labels: encoded labels
        padding_mask: padding mask
        logits: model logits (not predictions)
        seq_descs_batch: sequence descriptions
        mapping_dict_to_class: label mapping
        model: model object (to access CRF)
        device: device
        valid_mask: valid positions mask
    """

    num_to_show = min(32, v_labels.shape[0])  # Show examples
    list_seq_types = ["none"]

    for seq_i in range(num_to_show):
        mask = ~padding_mask[seq_i]

        if seq_descs_batch[seq_i] not in list_seq_types:
            print("Sequence type:", seq_descs_batch[seq_i], flush=True)

            # Get labels for this sequence
            labels_masked = v_labels[seq_i][mask].cpu().numpy().astype(int)

            # Decode prediction for sequence
            seq_logits = logits[seq_i : seq_i + 1]  # [1, seq_len, num_classes]
            seq_valid_mask = valid_mask[seq_i : seq_i + 1]  # [1, seq_len]

            pred_decoded = model.CRF.crf.decode(seq_logits, mask=seq_valid_mask)
            preds_masked = torch.tensor(pred_decoded[0], dtype=torch.long, device=device).cpu().numpy().astype(int)

            # Convert labels to RF vectors
            labels_rf = [mapping_dict_to_class[label] for label in labels_masked]

            # Convert predictions to RF vectors
            preds_rf = [mapping_dict_to_class[pred] for pred in preds_masked]

            # Extract individual RF sequences
            labels_rf0 = [rf[0] for rf in labels_rf]
            labels_rf1 = [rf[1] for rf in labels_rf]
            labels_rf2 = [rf[2] for rf in labels_rf]
            preds_rf0 = [rf[0] for rf in preds_rf]
            preds_rf1 = [rf[1] for rf in preds_rf]
            preds_rf2 = [rf[2] for rf in preds_rf]

            print("Encoded labels and preds:", flush=True)
            print(v_labels[seq_i][mask].cpu().numpy().astype(float), flush=True)
            print(preds_masked.astype(float), flush=True)
            print(flush=True)
            print("Labels RF0:", labels_rf0, flush=True)
            print("Labels RF1:", labels_rf1, flush=True)
            print("Labels RF2:", labels_rf2, flush=True)
            print(flush=True)
            print("Predictions RF0:", preds_rf0, flush=True)
            print("Predictions RF1:", preds_rf1, flush=True)
            print("Predictions RF2:", preds_rf2, flush=True)
            print("\n", flush=True)

        if seq_i == 3:
            # Only show a few of "easy-to-classify" samples
            list_seq_types = ["non-coding", "coding", "coding_with_substitutions"]


######################################################################################################################################################################################################
############################################################################################## Main ##################################################################################################
######################################################################################################################################################################################################

# Set seed for reproducibility
set_seed(args.seed)
print(f"Using random seed: {args.seed}", flush=True)

# Load in data once
print("Processing data from CSV files", flush=True)
train_data, val_data, sequence_types, seq_type_desc_fracs = load_and_process_data(max_len, dataset_size)

# Load the optimized hyperparameters
cfg = OmegaConf.load(f"{input_data_dir_path}/hyperparameter_configs/full_model_hyperparameters.yaml")

# Access them
act_function = cfg.hyperparameters.act_function
depth_transformer_encoder_blocks = cfg.hyperparameters.depth_transformer_encoder_blocks
dropout_rate_1 = cfg.hyperparameters.dropout_rate_1
dropout_rate_2 = cfg.hyperparameters.dropout_rate_2
lr_esm2 = cfg.hyperparameters.lr_esm2
lr_scratch = cfg.hyperparameters.lr_scratch
n_attention_heads = cfg.hyperparameters.n_attention_heads
transition_weight = cfg.hyperparameters.transition_weight

batch_size = default_train_batch_size
val_batch_size = default_val_batch_size

# Check for wandb run ID if resuming
wandb_run_id = None
checkpoint_path = f"{models_output_dir_path}/checkpoint_{dataset_size}_seed_{args.seed}{model_checkpoint_extension}.pt"
if args.resume and os.path.exists(checkpoint_path):
    # Load only the wandb run ID from checkpoint
    checkpoint_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    wandb_run_id = checkpoint_meta.get("wandb_run_id")
    del checkpoint_meta
    if wandb_run_id:
        print(f"Resuming WandB run: {wandb_run_id}", flush=True)

wandb.init(
    project=wandb_project_name,
    config={
        "seed": args.seed,
        "esm_model": args.esm_model,
        "gradient_checkpointing": gradient_checkpointing,
        "batch_size": batch_size,
        "depth_transformer_encoder_blocks": depth_transformer_encoder_blocks,
        "n_attention_heads": n_attention_heads,
        "dropout_rate_1": dropout_rate_1,
        "dropout_rate_2": dropout_rate_2,
        "act_function": act_function,
        "transition_weight": transition_weight,
        "lr_esm2": lr_esm2,
        "lr_scratch": lr_scratch,
    },
    name=f"train_{dataset_size}_{args.esm_model}_seed_{args.seed}",
    id=wandb_run_id,
    resume="must" if wandb_run_id else None,
)

dataloader_num_workers = 2

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=dataloader_num_workers,
    pin_memory=pin_memory,
    drop_last=True
)

val_loader = DataLoader(
    val_data,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=dataloader_num_workers,
    pin_memory=pin_memory
)


# Initialize model with hyperparameters to try
model, mapping_dict_to_class = initialize_model(
    device,
    num_layers=depth_transformer_encoder_blocks,
    n_attention_heads=n_attention_heads,
    dropout_rate_1=dropout_rate_1,
    dropout_rate_2=dropout_rate_2,
    act_function=act_function,
    transition_weight=transition_weight,
    label_classes=label_classes,
    use_gradient_checkpointing=gradient_checkpointing,
)

#MODIFY
print(f"\n=== Configuration Check ===", flush=True)
print(f"Error type: {args.error_type}", flush=True)
print(f"Label classes: {label_classes}", flush=True)
print(f"Number of encoded labels (CRF tags): {len(mapping_dict_to_class)}", flush=True)
print(f"Legal transitions defined for states: {list(model.CRF.legal_transitions.keys())}", flush=True)
print(f"Legal transitions: {model.CRF.legal_transitions}", flush=True)
print(f"Biologically valid transitions: {model.CRF.biologically_valid_mask.sum().item()}", flush=True)
print(f"Illegal transitions: {(~model.CRF.biologically_valid_mask).sum().item()}", flush=True)
print(f"Frequent self-transitions: {model.CRF.frequent_transition_mask.sum().item()}", flush=True)
print(f"=== End Configuration Check ===\n", flush=True)

# Prepare for unfreezing later (but don't unfreeze yet!)
model.sequence_encoder.prepare_for_unfreezing(unfreeze_fraction=0.5)

# Define settings for training
epochs = 20
steps_per_epoch = len(train_data) / batch_size
print("Steps per epoch: ", steps_per_epoch, flush=True)
eval_every_n_steps = steps_between_vals  #Validate every 10000 batches = 10000 * 32 samples #MODIFY
print(f"Evaluating {round(steps_per_epoch / eval_every_n_steps, 1)} times per epoch", flush=True)
freeze_esm_validations = int(3000000 / batch_size / eval_every_n_steps)  # unfreeze after having seen 3 million samples # MODIFY

# Initialize the loss tracker
tracker = CategoricalLossTracker(sequence_types)

# Initialize optimizer - NO ESM-2 parameters at all initially
optimizer = torch.optim.AdamW(
    [
        {"params": model.TransformerEncoderBlock.parameters(), "lr": lr_scratch},
        {"params": model.linear_transform.parameters(), "lr": lr_scratch},
        {"params": model.CRF.parameters(), "lr": lr_scratch},
    ]
)

# Initialize learning rate scheduler - reduces LR when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',           # Minimize validation loss
    factor=0.7,           # Reduce LR by 30% when plateau detected
    patience=6,           # Wait 6 validations before reducing
    min_lr=1e-7           # Don't reduce below this
)

# Store initial LR for tracking scheduler changes (use first param group as reference)
initial_lr = optimizer.param_groups[0]["lr"]

# Initialize variables for early stopping
best_val_loss = float("inf")

threshold_patience = 16
counter_patience = 0

# Initialize variables for training loop
step = 0
val_times_counter = 0

# Initialize warmup state (will be set when unfreezing)
warmup_state = None

# Initialize mixed precision scaler if using CUDA
scaler = GradScaler() if "cuda" in device_type else None
if scaler is not None:
    print("Mixed precision training enabled", flush=True)

# Initialize esm2_initial_params (will be set when unfreezing)
esm2_initial_params = None

# Load checkpoint if resuming (checkpoint_path defined earlier before wandb.init)
start_epoch = 0
if args.resume:
    if os.path.exists(checkpoint_path):
        restored_state = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler, device
        )
        start_epoch = restored_state["epoch"]
        step = restored_state["step"]
        val_times_counter = restored_state["val_times_counter"]
        best_val_loss = restored_state["best_val_loss"]
        counter_patience = restored_state["counter_patience"]
        warmup_state = restored_state["warmup_state"]
        esm2_initial_params = restored_state["esm2_initial_params"]
    else:
        print(f"Warning: --resume specified but no checkpoint found at {checkpoint_path}", flush=True)
        print("Starting training from scratch.", flush=True)

train_losses = [] #MODIFY (Before epoch loss, adapt)

# Training loop
for epoch in range(start_epoch, epochs):
    model.train()

    for i, batch in enumerate(train_loader):
        # Run training iteration with warmup state
        train_losses, warmup_state = training_iteration(i, batch, scaler, model, optimizer, device, train_losses, warmup_state)

        # Validation loop
        if step % eval_every_n_steps == 0 and step > 0:
            print("Eval starting...", flush=True)
            clear_memory()
            model.eval()

            # Initialize
            val_losses = []
            all_val_true_sequences = []
            all_val_pred_sequences = []
            all_val_sequence_types = []

            with torch.no_grad():

                tracker = CategoricalLossTracker(sequence_types) #MODIFY

                # DEBUG: memory at start of validation
                if torch.cuda.is_available():
                    print(f"[MEM] Val start | allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB", flush=True)

                for counter, val_batch in enumerate(val_loader):
                    v_inputs_nt_rf0 = val_batch["nt_encodings_rf0"].to(device, non_blocking=True)
                    v_inputs_aa_rf0 = val_batch["aa_encodings_rf0"]["input_ids"].to(device, non_blocking=True)
                    v_attention_mask_aa_rf0 = val_batch["aa_encodings_rf0"]["attention_mask"].to(device, non_blocking=True)

                    v_inputs_nt_rf1 = val_batch["nt_encodings_rf1"].to(device, non_blocking=True)
                    v_inputs_aa_rf1 = val_batch["aa_encodings_rf1"]["input_ids"].to(device, non_blocking=True)
                    v_attention_mask_aa_rf1 = val_batch["aa_encodings_rf1"]["attention_mask"].to(device, non_blocking=True)

                    v_inputs_nt_rf2 = val_batch["nt_encodings_rf2"].to(device, non_blocking=True)
                    v_inputs_aa_rf2 = val_batch["aa_encodings_rf2"]["input_ids"].to(device, non_blocking=True)
                    v_attention_mask_aa_rf2 = val_batch["aa_encodings_rf2"]["attention_mask"].to(device, non_blocking=True)

                    v_encoded_labels = val_batch["label_encodings"].to(device, non_blocking=True).long()

                    # Save padding mask before overwriting v_labels
                    padding_mask = v_encoded_labels == -1
                    valid_mask = ~padding_mask

                    # Forward pass for validation
                    if scaler is not None:
                        with autocast("cuda"):
                            v_outputs = model(
                                v_inputs_nt_rf0,
                                v_inputs_aa_rf0,
                                v_attention_mask_aa_rf0,
                                v_inputs_nt_rf1,
                                v_inputs_aa_rf1,
                                v_attention_mask_aa_rf1,
                                v_inputs_nt_rf2,
                                v_inputs_aa_rf2,
                                v_attention_mask_aa_rf2,
                                v_encoded_labels,
                            )
                            v_loss = v_outputs["loss"]
                    else:
                        v_outputs = model(
                            v_inputs_nt_rf0,
                            v_inputs_aa_rf0,
                            v_attention_mask_aa_rf0,
                            v_inputs_nt_rf1,
                            v_inputs_aa_rf1,
                            v_attention_mask_aa_rf1,
                            v_inputs_nt_rf2,
                            v_inputs_aa_rf2,
                            v_attention_mask_aa_rf2,
                            v_encoded_labels,
                        )
                        v_loss = v_outputs["loss"]

                    val_losses.append(v_loss.item())

                    # DEBUG: memory after forward pass
                    if counter < 5 and torch.cuda.is_available():
                        print(f"[MEM] Val batch {counter} (post-forward) | allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB", flush=True)

                    # Calculate category-specific losses
                    seq_descs_batch = val_batch["seq_desc"]

                    for desc in tracker.categories: #MODIFY
                        desc_mask = torch.tensor([d == desc for d in seq_descs_batch], device=device)
                        if desc_mask.any():
                            desc_count = desc_mask.sum().item()
                            desc_logits = v_outputs["logits"][desc_mask]
                            desc_labels_original = v_encoded_labels[desc_mask]
                            desc_valid_mask = valid_mask[desc_mask]

                            # Make labels safe - replace -1 with 0 (same as in LinearChainCRF.forward)
                            safe_desc_labels = desc_labels_original.clone()
                            safe_desc_labels[safe_desc_labels == -1] = 0

                            desc_ll = model.CRF.crf(desc_logits, safe_desc_labels, mask=desc_valid_mask, reduction="none")
                            desc_crf_loss = -desc_ll.mean()  # Mean over sequences, not tokens

                            tracker.update(desc, desc_crf_loss.item(), desc_count)

                            del desc_logits, desc_labels_original, safe_desc_labels, desc_valid_mask, desc_crf_loss
                        del desc_mask
                    
                    if counter % 30 == 0:
                        del v_inputs_nt_rf0, v_inputs_aa_rf0, v_attention_mask_aa_rf0
                        del v_inputs_nt_rf1, v_inputs_aa_rf1, v_attention_mask_aa_rf1
                        del (
                            v_inputs_nt_rf2,
                            v_inputs_aa_rf2,
                            v_attention_mask_aa_rf2,
                        )
                        del v_outputs, v_loss
                        clear_memory()
                        # DEBUG: memory after cleanup
                        if torch.cuda.is_available():
                            print(f"[MEM] Val batch {counter} (post-cleanup) | allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB", flush=True)

            # Calculate validation metrics
            if val_losses:
                val_avg_loss = sum(val_losses) / len(val_losses)
            else:
                val_avg_loss = float("inf")

            # After validation loop, verify losses match MODIFY; ADD
            weighted_sum = sum(
                seq_type_desc_fracs[cat] * tracker.get_metrics()[cat] 
                for cat in tracker.categories
            )
            print(f"val_avg_loss: {val_avg_loss:.4f}", flush = True)
            print(f"weighted sum from categories: {weighted_sum:.4f}", flush = True)

            # Calculate performance metrics
            if all_val_true_sequences and all_val_pred_sequences:
                sequence_metrics = calculate_sequence_accuracy_metrics(all_val_true_sequences, all_val_pred_sequences, all_val_sequence_types)
            else: #MODIFY; ADD
                sequence_metrics = {}

            # Calculate training loss MODIFY
            if train_losses:
                train_avg_loss = sum(train_losses) / len(train_losses)
            else:
                train_avg_loss = float("inf")

            # Early stopping check
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                torch.save(model.state_dict(), f"{models_output_dir_path}/full_model_{dataset_size}_seed_{args.seed}_trained{model_checkpoint_extension}.pth")  # modify
                counter_patience = 0
            else:
                counter_patience += 1

            # Save checkpoint after every validation for crash recovery
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                step,
                val_times_counter,
                best_val_loss,
                counter_patience,
                warmup_state,
                esm2_initial_params,
                wandb.run.id,
            )

            if counter_patience >= threshold_patience:
                print("Early stopping triggered!", flush=True)
                break

            # Step the learning rate scheduler based on validation loss
            scheduler.step(val_avg_loss)

            # Track LR as percentage of initial (starts at 100%, decreases when scheduler triggers)
            current_lr = optimizer.param_groups[0]["lr"]
            lr_percentage = (current_lr / initial_lr) * 100

            # Get evaluation metrics (returns counter and wandb dict)
            val_times_counter, wandb_metrics = log_evaluation_metrics(
                epoch,
                train_avg_loss,
                val_avg_loss,
                best_val_loss,
                tracker,
                sequence_metrics,
                val_times_counter,
                sequence_types,
            )

            # Consolidate all metrics for single wandb.log call
            all_metrics = {**wandb_metrics, "lr_percent_of_initial": lr_percentage}

            if val_times_counter == freeze_esm_validations:
                # Unfreeze ESM-2 layers and set up warmup
                esm2_param_group_idx = model.sequence_encoder.unfreeze_top_layers(lr_esm2=lr_esm2, optimizer=optimizer, warmup_factor=0.1)  # Start at 10% of lr_esm2

                # Initialize warmup state that will be used in training_iteration
                warmup_state = {
                    "counter": 0,
                    "total_steps": 1000,  # Warmup over 1000 steps
                    "param_group_idx": esm2_param_group_idx,
                    "target_lr": lr_esm2,
                }

                # Save initial ESM-2 parameters when unfreezing
                esm2_initial_params = {name: param.clone().detach() for name, param in model.sequence_encoder.pretrained_model_aa.named_parameters() if param.requires_grad}

            # During validation (every few validations after unfreezing)
            if val_times_counter > freeze_esm_validations and esm2_initial_params is not None:
                # Check how much parameters have changed
                total_change = 0
                total_norm = 0

                for name, param in model.sequence_encoder.pretrained_model_aa.named_parameters():
                    if name in esm2_initial_params:
                        change = (param - esm2_initial_params[name]).norm().item()
                        norm = esm2_initial_params[name].norm().item()
                        total_change += change**2
                        total_norm += norm**2

                total_change = total_change**0.5
                total_norm = total_norm**0.5
                relative_change = total_change / total_norm

                print(f"ESM-2 parameters changed by: {relative_change:.6e} ({relative_change * 100:.4f}%)", flush=True)

                # Add ESM-2 metrics to consolidated log
                all_metrics["esm2_cumulative_change"] = relative_change
                all_metrics["validations_since_unfreeze"] = val_times_counter - freeze_esm_validations

            # Single consolidated wandb.log call
            wandb.log(all_metrics)

            # Clean up validation data
            del val_losses, all_val_true_sequences, all_val_pred_sequences, all_val_sequence_types
            clear_memory()

            train_losses = []  # Reset training losses after validation MODIFY
            model.train()  # Back to training mode

        # Increment global step counter after each batch
        step += 1

    # Check for early stopping again at end of epoch
    if counter_patience >= threshold_patience:
        break

wandb.finish()
