"""
Train DeepCDS model using ONLY codon/nucleotide (one-hot) encodings.

Differences from train_full_model.py:
- No ESM-2 / protein language model input
- Sinusoidal positional encoding + linear projection replace the pLM embedding
- No amino acid tokenization in the data pipeline
- Single learning rate for all parameters (no staged ESM-2 unfreezing)

New CLI argument:
  --d_model   Hidden dimension for the Transformer encoder.
              Must be divisible by n_attention_heads from the hyperparameter
              config (default: 128).
"""

import gc
import math
import os
import pickle
import random

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

torch.cuda.empty_cache()
pd.options.mode.chained_assignment = None

import argparse

parser = argparse.ArgumentParser(description="Train DeepCDS Codon-Only Model")
parser.add_argument(
    "--dataset_size",
    type=str,
    default="all_genomes",
    choices=["100_genomes", "200_genomes", "400_genomes", "all_genomes"],
    help="Dataset size to use for training (default: all_genomes)",
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
    "--d_model",
    type=int,
    default=332,
    help="Hidden dimension for the Transformer encoder. Must be divisible by n_attention_heads from the hyperparameter config (default: 332).",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug mode with smaller dataset and more frequent validation",
)
parser.add_argument(
    "--compute_sequence_metrics",
    action="store_true",
    help="Compute per-sequence accuracy metrics (MCC, perfect sequences) during validation. Disabled by default as it is time-consuming.",
)

args = parser.parse_args()

# Set variables based on error type
if args.error_type == "indel_substitution":
    model_dir_path_suffix = "model_with_errors"
    label_classes = 6
    wandb_project_name = "DeepCDS_errors_codon_only"
    steps_between_vals = 10000
elif args.error_type == "substitution":
    model_dir_path_suffix = "model_with_substitution_errors"
    label_classes = 4
    wandb_project_name = "DeepCDS_substitution_errors_codon_only"
    steps_between_vals = 6000
else:
    model_dir_path_suffix = "model_without_errors"
    label_classes = 4
    wandb_project_name = "DeepCDS_no_errors_codon_only"
    steps_between_vals = 5000

model_checkpoint_extension = "_final"

if args.debug:
    steps_between_vals = 100
    frac_train = 0.05
    frac_val = 0.01
    model_checkpoint_extension = "_debug"
    wandb_project_name = "debug"
else:
    frac_train = 1.0
    frac_val = 0.25

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if args.healthtech_cluster:
    input_data_dir_path = f"/net/well/pool/projects2/lisani/DeepCDS/FragmentPredictor/data/processed_data/model_data/shared_crf/{model_dir_path_suffix}"
    num_workers_cpu = 2
    pin_memory = True
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device_type = device.type
    print("Device: ", device, flush=True)
    assert device == torch.device("cuda"), "HealthTech cluster run should be on a CUDA GPU."
    os.makedirs(f"{input_data_dir_path}/models/", exist_ok=True)
    models_output_dir_path = f"{input_data_dir_path}/models/"

elif args.scarb_cluster:
    input_data_dir_path = f"/tmp/nrt204/FragmentPredictor/data/processed_data/model_data/shared_crf/{model_dir_path_suffix}"
    num_workers_cpu = 2
    pin_memory = True
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device_type = device.type
    print("Device: ", device, flush=True)
    assert device.type == "cuda", "SCARB cluster run should be on a CUDA GPU."
    _cuda_mem_limit_gb = 48
    _total_vram = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(_cuda_mem_limit_gb * 1e9 / _total_vram, device)
    print(f"CUDA memory capped at {_cuda_mem_limit_gb} GB / {_total_vram / 1e9:.1f} GB total", flush=True)
    os.makedirs(f"{input_data_dir_path}/models/", exist_ok=True)
    models_output_dir_path = f"{input_data_dir_path}/models/"

else:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device_type = device.type
    input_data_dir_path = f"../../../data/processed_data/model_data/shared_crf/{model_dir_path_suffix}"
    num_workers_cpu = 0
    pin_memory = False
    print(f"Device type: {device_type}, GPU: {args.gpu if device_type == 'cuda' else 'N/A'}", flush=True)
    os.makedirs(f"{input_data_dir_path}/models/", exist_ok=True)
    models_output_dir_path = f"{input_data_dir_path}/models/"

default_train_batch_size = 32
default_val_batch_size = 480

dataset_size = args.dataset_size
max_aa_len = 100  # Number of codons per window


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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def one_hot_encode(sequence):
    """
    One-hot encode a nucleotide sequence.

    Args:
        sequence (str): Nucleotide sequence.

    Returns:
        torch.Tensor: Shape (4, len(sequence)).
    """
    seq_len = len(sequence)
    encoding = torch.zeros(4, seq_len, dtype=torch.float32)
    seq_bytes = torch.frombuffer(bytearray(sequence.encode("ascii")), dtype=torch.uint8)
    for nuc, idx in ((65, 0), (67, 1), (71, 2), (84, 3)):
        mask = seq_bytes == nuc
        encoding[idx, mask] = 1.0
    return encoding


def process_nt_sequences_to_codons(nt_sequences, max_aa_len):
    """
    Convert nucleotide one-hot tensors of shape (4, nt_len) to codon tensors of shape (max_aa_len, 12)
    by grouping every 3 nucleotides into a single 12-dim vector.
    """
    stacked = torch.stack(nt_sequences)  # (batch, 4, nt_len)
    batch_size = stacked.shape[0]
    reshaped = stacked.view(batch_size, 4, max_aa_len, 3)
    transposed = reshaped.permute(0, 2, 1, 3)  # (batch, max_aa_len, 4, 3)
    formatted = transposed.reshape(batch_size, max_aa_len, 12)
    return list(formatted.unbind(0))


class SeqDataset(torch.utils.data.Dataset):
    """
    Dataset for the codon-only model. Stores nucleotide codon encodings and
    labels for all three reading frames. No amino acid encodings.

    Args:
        nt_encodings_rf{0,1,2} (np.ndarray): Codon encodings of shape (N_samples, max_aa_len, 12).
        labels_rf{0,1,2} (np.ndarray): Per-position per-RF labels of shape (N_samples, max_aa_len).
        label_encodings (np.ndarray): Shared CRF label sequence of shape (N_samples, max_aa_len).
        seq_desc (list[str]): Sequence description strings.
    """

    def __init__(
        self,
        nt_encodings_rf0,
        labels_rf0,
        nt_encodings_rf1,
        labels_rf1,
        nt_encodings_rf2,
        labels_rf2,
        label_encodings,
        seq_desc,
    ):
        self.nt_encodings_rf0 = nt_encodings_rf0
        self.labels_rf0 = labels_rf0
        self.nt_encodings_rf1 = nt_encodings_rf1
        self.labels_rf1 = labels_rf1
        self.nt_encodings_rf2 = nt_encodings_rf2
        self.labels_rf2 = labels_rf2
        self.label_encodings = label_encodings
        self.seq_desc = seq_desc

    def __getitem__(self, idx):
        def _to_tensor(x):
            return torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.tensor(x, dtype=torch.float32)

        return {
            "nt_encodings_rf0": torch.as_tensor(self.nt_encodings_rf0[idx], dtype=torch.float32),
            "labels_rf0": _to_tensor(self.labels_rf0[idx]),
            "nt_encodings_rf1": torch.as_tensor(self.nt_encodings_rf1[idx], dtype=torch.float32),
            "labels_rf1": _to_tensor(self.labels_rf1[idx]),
            "nt_encodings_rf2": torch.as_tensor(self.nt_encodings_rf2[idx], dtype=torch.float32),
            "labels_rf2": _to_tensor(self.labels_rf2[idx]),
            "label_encodings": _to_tensor(self.label_encodings[idx]),
            "seq_desc": self.seq_desc[idx],
        }

    def __len__(self):
        return len(self.label_encodings)


def encode_data(processed_samples_df, max_aa_len=max_aa_len):
    """
    Encode data samples using only nucleotide/codon encodings. No amino acid tokenization.

    Args:
        processed_samples_df (DataFrame): Must contain columns rf{0,1,2}_seq_nt,
            rf{0,1,2}_labels, label_encodings, and seq_desc.
        max_aa_len (int): Maximum number of codons per window.

    Returns:
        dataset (SeqDataset): Encoded dataset.
        sequence_types (list[str]): Unique sequence description strings.
    """
    encodings_nt = {}
    labels = {}
    max_nt_len = max_aa_len * 3

    # Shared (CRF) label sequence
    if isinstance(processed_samples_df["label_encodings"].iloc[0], str):
        processed_samples_df["label_encodings"] = processed_samples_df["label_encodings"].apply(eval)

    label_arrays = [np.array(x, dtype=np.int8) for x in processed_samples_df["label_encodings"]]
    padded_labels = np.full((len(label_arrays), max_aa_len), -1, dtype=np.int8)
    for i, arr in enumerate(label_arrays):
        length = min(len(arr), max_aa_len)
        padded_labels[i, :length] = arr[:length]

    for rf in ["rf0", "rf1", "rf2"]:
        # Per-RF labels
        if isinstance(processed_samples_df[f"{rf}_labels"].iloc[0], str):
            processed_samples_df[f"{rf}_labels"] = processed_samples_df[f"{rf}_labels"].apply(eval)

        raw_labels = [np.array(x, dtype=np.int8) for x in processed_samples_df[f"{rf}_labels"]]
        padded_rf = np.full((len(raw_labels), max_aa_len), -1, dtype=np.int8)
        for i, arr in enumerate(raw_labels):
            length = min(len(arr), max_aa_len)
            padded_rf[i, :length] = arr[:length]
        labels[rf] = padded_rf
        del raw_labels

        # Nucleotide sequence: pad short sequences with 'N' (→ all-zero codon = padding)
        processed_samples_df[f"{rf}_seq_nt"] = processed_samples_df[f"{rf}_seq_nt"].apply(
            lambda seq: seq + "N" * (max_nt_len - len(seq)) if len(seq) < max_nt_len else seq
        )

        nt_sequences = [one_hot_encode(seq) for seq in processed_samples_df[f"{rf}_seq_nt"]]
        nt_encodings_rf = process_nt_sequences_to_codons(nt_sequences, max_aa_len)
        encodings_nt[rf] = np.array([t.numpy() for t in nt_encodings_rf])
        del nt_sequences, nt_encodings_rf
        gc.collect()

    seq_descriptions = processed_samples_df["seq_desc"].tolist()

    dataset = SeqDataset(
        encodings_nt["rf0"], labels["rf0"],
        encodings_nt["rf1"], labels["rf1"],
        encodings_nt["rf2"], labels["rf2"],
        padded_labels,
        seq_descriptions,
    )

    return dataset, list(set(seq_descriptions))


def load_and_process_data(dataset_size):
    """Load CSVs, subsample, and encode to codon tensors."""
    train_set = pd.read_csv(
        f"{input_data_dir_path}/datasets_model/train_{dataset_size}.csv.gz",
        index_col=None,
        compression="gzip",
    )
    val_set = pd.read_csv(f"{input_data_dir_path}/datasets_model/val.csv.gz", index_col=None, compression="gzip")

    val_set["accession_seq_desc_merged"] = val_set["accession"].astype(str) + "_" + val_set["seq_desc"].astype(str)
    val_set = val_set.groupby("accession_seq_desc_merged", group_keys=False).apply(
        lambda x: x.sample(frac=frac_val, random_state=42)
    )

    train_set["accession_seq_desc_merged"] = train_set["accession"].astype(str) + "_" + train_set["seq_desc"].astype(str)
    train_set = train_set.groupby("accession_seq_desc_merged", group_keys=False).apply(
        lambda x: x.sample(frac=frac_train, random_state=42)
    )

    seq_type_desc_fracs = (val_set["seq_desc"].value_counts(normalize=True)).to_dict()
    seq_type_desc_fracs_train = (train_set["seq_desc"].value_counts(normalize=True)).to_dict()

    print("Training data samples: ", train_set.shape[0], flush=True)
    print("Validation data samples during training: ", val_set.shape[0], flush=True)
    print("Distribution of sequence types in training set:", seq_type_desc_fracs_train, flush=True)
    print("Distribution of sequence types in validation set:", seq_type_desc_fracs, flush=True)

    train_data, sequence_types = encode_data(train_set)
    del train_set
    gc.collect()

    val_data, _ = encode_data(val_set)
    del val_set
    gc.collect()

    return train_data, val_data, sequence_types, seq_type_desc_fracs


######################################################################################################################################################################################################
################################################################################################Model####################################################################################################
######################################################################################################################################################################################################


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Args:
        d_model (int): Embedding dimension.
        max_len (int): Maximum sequence length (number of codons).
        dropout (float): Dropout applied after adding the encoding.
    """

    def __init__(self, d_model, max_len, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, N, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


class CodonTransformerEncoderBlock(nn.Module):
    """
    Transformer encoder that operates solely on codon one-hot encodings.

    Pipeline per reading frame:
        codon one-hot (B, N, 12)
        → linear projection (B, N, d_model)
        → sinusoidal positional encoding
        → Transformer encoder
        → LayerNorm
        → linear classifier (B, N, num_labels)

    Args:
        d_model (int): Hidden dimension (must be divisible by n_attention_heads).
        num_layers (int): Number of Transformer encoder layers.
        n_attention_heads (int): Number of attention heads.
        dropout_rate (float): Dropout rate inside the Transformer and positional encoding.
        act_function (str): Feedforward activation ('relu' or 'gelu').
        num_labels (int): Number of per-RF output classes.
        max_len (int): Maximum sequence length in codons.
    """

    def __init__(self, d_model, num_layers, n_attention_heads, dropout_rate, act_function, num_labels, max_len):
        super().__init__()

        self.input_projection = nn.Linear(12, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout=dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_attention_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout_rate,
            activation=act_function,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, num_labels)

    def forward(self, encoded_seqs_nt, attention_mask):
        """
        Args:
            encoded_seqs_nt (torch.Tensor): Codon one-hot encodings (B, N, 12).
            attention_mask (torch.Tensor): Boolean mask (B, N), True = valid codon.

        Returns:
            torch.Tensor: Logits (B, N, num_labels).
        """
        x = self.input_projection(encoded_seqs_nt)  # (B, N, d_model)
        x = self.pos_encoding(x)                    # (B, N, d_model)

        x = x.permute(1, 0, 2)                                        # (N, B, d_model)
        x = self.encoder(x, src_key_padding_mask=~attention_mask)      # (N, B, d_model)
        x = x.permute(1, 0, 2)                                        # (B, N, d_model)

        x = self.norm(x)
        return self.linear(x)  # (B, N, num_labels)


class LinearChainCRF(nn.Module):
    """
    CRF layer with biologically constrained transition initialisation.

    Args:
        mapping_dict_to_class (dict): Maps integer label indices to (rf0, rf1, rf2) tuples.
        transition_weight (float): Initial weight for legal-but-infrequent transitions.
        num_encoded_labels (int, optional): Total number of labels; inferred if None.
        label_classes (int): Number of per-frame label classes (4 or 6).
    """

    def __init__(self, mapping_dict_to_class, transition_weight, num_encoded_labels=None, label_classes=label_classes):
        super().__init__()

        self.shared_rf_labels_mapping = mapping_dict_to_class

        if num_encoded_labels is None:
            num_encoded_labels = len(self.shared_rf_labels_mapping)

        self.crf = CRF(num_tags=num_encoded_labels, batch_first=True)

        if label_classes == 6:
            self.legal_transitions = {
                0: {0, 2, 4}, 1: {1, 3, 5}, 2: {1, 5},
                3: {0, 2, 4}, 4: {1, 3, 5}, 5: {0, 2, 4},
            }
        elif label_classes == 4:
            self.legal_transitions = {
                0: {0, 2},
                1: {1, 3},
                2: {1},
                3: {0, 2},
            }
        else:
            raise ValueError("label_classes must be 4 or 6.")

        self.biologically_valid_mask = torch.ones_like(self.crf.transitions, dtype=torch.bool)
        self.frequent_transition_mask = torch.zeros_like(self.crf.transitions, dtype=torch.bool)

        self._create_biologically_valid_mask()
        self._create_frequent_transition_mask()

        with torch.no_grad():
            self.crf.transitions[~self.biologically_valid_mask] = -10
            legal_infrequent = self.biologically_valid_mask & ~self.frequent_transition_mask
            self.crf.transitions[legal_infrequent] = transition_weight
            self.crf.transitions[self.frequent_transition_mask] = 0

    def _is_legal_transition(self, from_rf, to_rf):
        from_rf0, from_rf1, from_rf2 = from_rf
        to_rf0, to_rf1, to_rf2 = to_rf
        return (
            to_rf0 in self.legal_transitions[from_rf0]
            and to_rf1 in self.legal_transitions[from_rf1]
            and to_rf2 in self.legal_transitions[from_rf2]
        )

    def _create_biologically_valid_mask(self):
        num_labels = len(self.shared_rf_labels_mapping)
        legal_count = illegal_count = 0
        for from_label in range(num_labels):
            for to_label in range(num_labels):
                from_rf = self.shared_rf_labels_mapping[from_label]
                to_rf = self.shared_rf_labels_mapping[to_label]
                if self._is_legal_transition(from_rf, to_rf):
                    self.biologically_valid_mask[from_label, to_label] = True
                    legal_count += 1
                else:
                    self.biologically_valid_mask[from_label, to_label] = False
                    illegal_count += 1
        print(f"Legal transitions: {legal_count}", flush=True)
        print(f"Illegal transitions: {illegal_count}", flush=True)
        print(f"Percentage legal: {legal_count / (legal_count + illegal_count) * 100:.1f}%", flush=True)

    def _create_frequent_transition_mask(self):
        frequent_rf_combinations = {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)}
        frequent_labels = [
            label_idx
            for label_idx, rf_combo in self.shared_rf_labels_mapping.items()
            if rf_combo in frequent_rf_combinations
        ]
        for label in frequent_labels:
            self.frequent_transition_mask[label, label] = True
        print(f"Frequent self-transitions: {len(frequent_labels)}", flush=True)

    def forward(self, logits, attention_mask, labels=None):
        """
        Args:
            logits (torch.Tensor): Emission scores (B, N, L).
            attention_mask (torch.Tensor): Boolean mask (B, N), True = valid.
            labels (torch.Tensor, optional): Gold labels (B, N) with -1 for padding.

        Returns:
            dict: {'loss', 'logits'} during training; {'predictions', 'logits'} during inference.
        """
        if labels is not None:
            crf_mask = labels != -1  # (B, N)
            safe_labels = labels.clone()
            safe_labels[safe_labels == -1] = 0
            log_likelihood = self.crf(logits, safe_labels, mask=crf_mask, reduction="none")
            return {"loss": -log_likelihood.mean(), "logits": logits}
        else:
            crf_mask = attention_mask.bool()
            predictions = self.crf.decode(logits, mask=crf_mask)
            return {"predictions": predictions, "logits": logits}


class CodonOnlyCDSPredictor(nn.Module):
    """
    Full CDS prediction model using only codon one-hot encodings.

    Three reading frames are processed by the same shared CodonTransformerEncoderBlock.
    Their per-frame logits are concatenated, projected to the shared CRF label space,
    and decoded with a LinearChainCRF.

    The attention mask is derived directly from the codon encodings: positions where all
    12 values are zero are treated as padding (result of 'N'-padded nucleotide sequences).

    Args:
        d_model (int): Hidden dimension for the Transformer.
        num_layers (int): Number of Transformer encoder layers.
        n_attention_heads (int): Number of attention heads.
        dropout_rate (float): Dropout rate.
        act_function (str): Feedforward activation function.
        transition_weight (float): Initial weight for legal-but-infrequent CRF transitions.
        num_encoded_labels (int): Total number of combined RF label states.
        encoded_labels_mapping (dict): Maps label indices to RF tuples.
        label_classes (int): Per-frame label classes (4 or 6).
        max_len (int): Maximum sequence length in codons.
    """

    def __init__(
        self,
        d_model,
        num_layers,
        n_attention_heads,
        dropout_rate,
        act_function,
        transition_weight,
        num_encoded_labels,
        encoded_labels_mapping,
        label_classes=label_classes,
        max_len=max_aa_len,
    ):
        super().__init__()

        self.CodonTransformerEncoderBlock = CodonTransformerEncoderBlock(
            d_model=d_model,
            num_layers=num_layers,
            n_attention_heads=n_attention_heads,
            dropout_rate=dropout_rate,
            act_function=act_function,
            num_labels=label_classes,
            max_len=max_len,
        )

        self.pre_crf_norm = nn.LayerNorm(3 * label_classes)
        self.linear_transform = nn.Linear(3 * label_classes, num_encoded_labels)
        self.CRF = LinearChainCRF(
            mapping_dict_to_class=encoded_labels_mapping,
            transition_weight=transition_weight,
            num_encoded_labels=num_encoded_labels,
            label_classes=label_classes,
        )

    def forward(
        self,
        encoded_seqs_nt_rf0,
        encoded_seqs_nt_rf1,
        encoded_seqs_nt_rf2,
        labels=None,
    ):
        """
        Args:
            encoded_seqs_nt_rf{0,1,2} (torch.Tensor): Codon one-hot encodings (B, N, 12).
            labels (torch.Tensor, optional): CRF target labels (B, N) with -1 for padding.

        Returns:
            dict: {'loss', 'logits'} or {'predictions', 'logits'}.
        """
        # Derive attention mask: valid positions have at least one non-zero codon value.
        # N-padded positions are all-zero → treated as padding.
        attention_mask = encoded_seqs_nt_rf0.sum(dim=-1) != 0  # (B, N)

        logits_rf0 = self.CodonTransformerEncoderBlock(encoded_seqs_nt_rf0, attention_mask)  # (B, N, C)
        logits_rf1 = self.CodonTransformerEncoderBlock(encoded_seqs_nt_rf1, attention_mask)  # (B, N, C)
        logits_rf2 = self.CodonTransformerEncoderBlock(encoded_seqs_nt_rf2, attention_mask)  # (B, N, C)

        combined = torch.cat([logits_rf0, logits_rf1, logits_rf2], dim=-1)       # (B, N, 3*C)
        logits_encoded = self.linear_transform(self.pre_crf_norm(combined))       # (B, N, L)

        return self.CRF(logits=logits_encoded, attention_mask=attention_mask, labels=labels)


######################################################################################################################################################################################################
##########################################################################################Utility functions###########################################################################################
######################################################################################################################################################################################################


def print_model_dimensions(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}", flush=True)


def count_parameters(model):
    total_params_learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}", flush=True)
    print(f"Total trainable parameters: {total_params_learnable:,}", flush=True)


def initialize_model(device, d_model, num_layers, n_attention_heads, dropout_rate, act_function, transition_weight, label_classes):
    """
    Instantiate and move the codon-only model to device.

    Returns:
        model (CodonOnlyCDSPredictor)
        mapping_dict_to_class (dict)
    """
    print("Running on: ", device, flush=True)

    with open(f"{input_data_dir_path}/label_mappings/mapping_to_3d_vector.pkl", "rb") as f:
        mapping_dict_to_class = pickle.load(f)

    num_encoded_labels = len(mapping_dict_to_class)
    print(f"Number of encoded label classes: {num_encoded_labels}", flush=True)

    assert d_model % n_attention_heads == 0, (
        f"d_model ({d_model}) must be divisible by n_attention_heads ({n_attention_heads}). "
        "Adjust --d_model or n_attention_heads in the hyperparameter config."
    )

    model = CodonOnlyCDSPredictor(
        d_model=d_model,
        num_layers=num_layers,
        n_attention_heads=n_attention_heads,
        dropout_rate=dropout_rate,
        act_function=act_function,
        transition_weight=transition_weight,
        num_encoded_labels=num_encoded_labels,
        encoded_labels_mapping=mapping_dict_to_class,
        label_classes=label_classes,
        max_len=max_aa_len,
    )
    model.to(device)

    if device.type == "cuda":
        print(f"Memory allocated after loading model: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB", flush=True)

    count_parameters(model)
    return model, mapping_dict_to_class


def calculate_sequence_accuracy_metrics(true_labels_list, predictions_list, sequence_types_list=None):
    """Calculate sequence-level MCC, accuracy, and per-type metrics."""
    if len(true_labels_list) != len(predictions_list):
        raise ValueError("Mismatch in number of sequences")

    total_sequences = len(true_labels_list)
    perfect_sequences = 0
    high_accuracy_sequences = 0
    sequence_accuracies = []
    all_true_labels = []
    all_predictions = []

    type_specific_data = {}
    if sequence_types_list is not None:
        for seq_type in set(sequence_types_list):
            type_specific_data[seq_type] = {
                "true_labels": [], "predictions": [], "sequence_accuracies": [],
                "perfect_sequences": 0, "high_accuracy_sequences": 0, "total_sequences": 0,
            }

    batch_size = 1000
    for i in range(0, total_sequences, batch_size):
        batch_true = true_labels_list[i: i + batch_size]
        batch_pred = predictions_list[i: i + batch_size]
        batch_types = sequence_types_list[i: i + batch_size] if sequence_types_list else None

        for idx, (true_labels, predictions) in enumerate(zip(batch_true, batch_pred)):
            true_labels = np.asarray(true_labels)
            predictions = np.asarray(predictions)

            all_true_labels.extend(true_labels.tolist())
            all_predictions.extend(predictions.tolist())

            accuracy = (true_labels == predictions).mean()
            sequence_accuracies.append(accuracy)

            if accuracy == 1.0:
                perfect_sequences += 1
            if accuracy > 0.9:
                high_accuracy_sequences += 1

            if batch_types is not None:
                seq_type = batch_types[idx]
                td = type_specific_data[seq_type]
                td["true_labels"].extend(true_labels.tolist())
                td["predictions"].extend(predictions.tolist())
                td["sequence_accuracies"].append(accuracy)
                td["total_sequences"] += 1
                if accuracy == 1.0:
                    td["perfect_sequences"] += 1
                if accuracy > 0.9:
                    td["high_accuracy_sequences"] += 1

    all_true_labels = np.array(all_true_labels)
    all_predictions = np.array(all_predictions)

    try:
        overall_mcc = matthews_corrcoef(all_true_labels, all_predictions)
        if np.isnan(overall_mcc):
            overall_mcc = 0.0
    except Exception:
        overall_mcc = 0.0

    results = {
        "fraction_perfect_sequences": perfect_sequences / total_sequences,
        "fraction_high_accuracy_sequences": high_accuracy_sequences / total_sequences,
        "overall_mcc": overall_mcc,
        "accuracy": np.mean(sequence_accuracies),
    }

    if sequence_types_list is not None:
        for seq_type, td in type_specific_data.items():
            if td["total_sequences"] > 0:
                try:
                    type_mcc = matthews_corrcoef(np.array(td["true_labels"]), np.array(td["predictions"]))
                    if np.isnan(type_mcc):
                        type_mcc = 0.0
                except Exception:
                    type_mcc = 0.0
                results[f"{seq_type}_mcc"] = type_mcc
                results[f"{seq_type}_accuracy"] = np.mean(td["sequence_accuracies"])
                results[f"{seq_type}_fraction_perfect"] = td["perfect_sequences"] / td["total_sequences"]
                results[f"{seq_type}_fraction_high_accuracy"] = td["high_accuracy_sequences"] / td["total_sequences"]

    return results


class CategoricalLossTracker:
    """Track per-sequence-type losses during training/validation."""

    def __init__(self, categories):
        self.categories = categories
        self.total_loss = {cat: 0.0 for cat in categories}
        self.total_count = {cat: 0 for cat in categories}

    def update(self, category, loss, count):
        self.total_loss[category] += loss * count
        self.total_count[category] += count

    def get_metrics(self):
        return {
            cat: self.total_loss[cat] / self.total_count[cat] if self.total_count[cat] > 0 else 0.0
            for cat in self.categories
        }


def log_evaluation_metrics(epoch, train_avg_loss, val_avg_loss, best_val_loss, tracker, sequence_metrics, val_times_counter, sequence_types, train_tracker=None):
    print_parts = [
        f"---Evaluation {val_times_counter}---\n",
        f"Train Loss: {train_avg_loss:.4f}\t\t",
        f"Val Loss: {val_avg_loss:.4f}\t\t",
        f"Overall MCC: {sequence_metrics.get('overall_mcc', 0):.4f}\t\t",
        f"Overall Accuracy: {sequence_metrics.get('accuracy', 0):.4f}\n",
    ]
    for seq_type in sequence_types:
        print_parts.append(f"Val Loss {seq_type}: {tracker.get_metrics().get(seq_type, 0):.4f}\t\t")
    if train_tracker is not None:
        print_parts.append("\n")
        for seq_type in sequence_types:
            print_parts.append(f"Train Loss {seq_type}: {train_tracker.get_metrics().get(seq_type, 0):.4f}\t\t")
    print_parts.append("\n")
    for seq_type in sequence_types:
        type_mcc = sequence_metrics.get(f"{seq_type}_mcc", 0)
        type_acc = sequence_metrics.get(f"{seq_type}_accuracy", 0)
        if type_mcc != 0 or type_acc != 0:
            print_parts.extend([f"MCC {seq_type}: {type_mcc:.4f}\t\t", f"Acc {seq_type}: {type_acc:.4f}\t\t"])
    print("".join(print_parts), flush=True)

    wandb_log = {
        "epoch": epoch + 1,
        "train_loss": train_avg_loss,
        "val_loss": val_avg_loss,
        "val_fraction_perfect_sequences": sequence_metrics.get("fraction_perfect_sequences", 0),
        "val_fraction_high_accuracy_sequences": sequence_metrics.get("fraction_high_accuracy_sequences", 0),
        "val_overall_mcc": sequence_metrics.get("overall_mcc", 0),
        "val_accuracy": sequence_metrics.get("accuracy", 0),
        "best_val_loss": best_val_loss,
    }
    for seq_type in sequence_types:
        wandb_log[f"val_loss_{seq_type}"] = tracker.get_metrics().get(seq_type, 0)
    if train_tracker is not None:
        for seq_type in sequence_types:
            wandb_log[f"train_loss_{seq_type}"] = train_tracker.get_metrics().get(seq_type, 0)
    for seq_type in sequence_types:
        for key, metric_key in [("val_mcc", f"{seq_type}_mcc"), ("val_accuracy", f"{seq_type}_accuracy"),
                                 ("val_fraction_perfect", f"{seq_type}_fraction_perfect"),
                                 ("val_fraction_high_accuracy", f"{seq_type}_fraction_high_accuracy")]:
            val = sequence_metrics.get(metric_key, 0)
            if val != 0:
                wandb_log[f"{key}_{seq_type}"] = val

    return val_times_counter + 1, wandb_log


def training_iteration(i, batch, scaler, model, optimizer, device, train_losses, train_tracker=None):
    """Single training step (forward + backward + optimizer)."""
    inputs_nt_rf0 = batch["nt_encodings_rf0"].to(device, non_blocking=True)
    inputs_nt_rf1 = batch["nt_encodings_rf1"].to(device, non_blocking=True)
    inputs_nt_rf2 = batch["nt_encodings_rf2"].to(device, non_blocking=True)
    encoded_labels = batch["label_encodings"].to(device, non_blocking=True).long()

    if scaler is not None:
        with autocast("cuda"):
            outputs = model(inputs_nt_rf0, inputs_nt_rf1, inputs_nt_rf2, labels=encoded_labels)
            loss = outputs["loss"]
    else:
        outputs = model(inputs_nt_rf0, inputs_nt_rf1, inputs_nt_rf2, labels=encoded_labels)
        loss = outputs["loss"]

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"WARNING: NaN/Inf loss detected at batch {i}! Skipping.", flush=True)
        return train_losses, None

    optimizer.zero_grad()

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    last_loss = loss.item()
    train_losses.append(last_loss)

    if train_tracker is not None:
        with torch.no_grad():
            seq_descs_batch = batch["seq_desc"]
            valid_mask = encoded_labels != -1
            for desc in train_tracker.categories:
                desc_mask = torch.tensor([d == desc for d in seq_descs_batch], device=device)
                if desc_mask.any():
                    desc_count = desc_mask.sum().item()
                    desc_logits = outputs["logits"][desc_mask]
                    desc_labels_orig = encoded_labels[desc_mask]
                    desc_valid = valid_mask[desc_mask]
                    safe_desc_labels = desc_labels_orig.clone()
                    safe_desc_labels[safe_desc_labels == -1] = 0
                    desc_ll = model.CRF.crf(desc_logits, safe_desc_labels, mask=desc_valid, reduction="none")
                    train_tracker.update(desc, (-desc_ll.mean()).item(), desc_count)
                    del desc_logits, desc_labels_orig, safe_desc_labels, desc_valid, desc_ll
                del desc_mask

    if i % 10000 == 0:
        print("CRF transition matrix sample (first 5x5):", flush=True)
        print(model.CRF.crf.transitions[:5, :5], flush=True)

    if i % 1000 == 0:
        del outputs, loss
        del inputs_nt_rf0, inputs_nt_rf1, inputs_nt_rf2, encoded_labels

    return train_losses, last_loss


######################################################################################################################################################################################################
############################################################################################## Main ##################################################################################################
######################################################################################################################################################################################################

set_seed(args.seed)
print(f"Using random seed: {args.seed}", flush=True)

print("Processing data from CSV files", flush=True)
train_data, val_data, sequence_types, seq_type_desc_fracs = load_and_process_data(dataset_size)
print(f"Training sequences (encoded): {len(train_data):,}", flush=True)
print(f"Validation sequences (encoded): {len(val_data):,}", flush=True)

# Load hyperparameters (shared config; d_model is overridden by --d_model)
cfg = OmegaConf.load(f"{input_data_dir_path}/hyperparameter_configs/full_model_hyperparameters.yaml")

act_function = cfg.hyperparameters.act_function
depth_transformer_encoder_blocks = cfg.hyperparameters.depth_transformer_encoder_blocks
dropout_rate_2 = cfg.hyperparameters.dropout_rate_2
lr_scratch = cfg.hyperparameters.lr_scratch
n_attention_heads = cfg.hyperparameters.n_attention_heads
transition_weight = cfg.hyperparameters.transition_weight

batch_size = default_train_batch_size
val_batch_size = default_val_batch_size

wandb.init(
    project=wandb_project_name,
    config={
        "seed": args.seed,
        "model": "codon_only",
        "d_model": args.d_model,
        "batch_size": batch_size,
        "depth_transformer_encoder_blocks": depth_transformer_encoder_blocks,
        "n_attention_heads": n_attention_heads,
        "dropout_rate": dropout_rate_2,
        "act_function": act_function,
        "transition_weight": transition_weight,
        "lr": lr_scratch,
    },
    name=f"train_codon_only_{dataset_size}_d{args.d_model}_seed_{args.seed}",
)

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers_cpu,
    pin_memory=pin_memory,
    drop_last=True,
)
val_loader = DataLoader(
    val_data,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=num_workers_cpu,
    pin_memory=pin_memory,
)

model, mapping_dict_to_class = initialize_model(
    device,
    d_model=args.d_model,
    num_layers=depth_transformer_encoder_blocks,
    n_attention_heads=n_attention_heads,
    dropout_rate=dropout_rate_2,
    act_function=act_function,
    transition_weight=transition_weight,
    label_classes=label_classes,
)

print(f"\n=== Configuration Check ===", flush=True)
print(f"Error type: {args.error_type}", flush=True)
print(f"Label classes: {label_classes}", flush=True)
print(f"d_model: {args.d_model}", flush=True)
print(f"Number of encoded labels (CRF tags): {len(mapping_dict_to_class)}", flush=True)
print(f"Legal transitions: {model.CRF.legal_transitions}", flush=True)
print(f"Biologically valid transitions: {model.CRF.biologically_valid_mask.sum().item()}", flush=True)
print(f"=== End Configuration Check ===\n", flush=True)

epochs = 20
steps_per_epoch = len(train_data) / batch_size
print("Steps per epoch: ", steps_per_epoch, flush=True)
eval_every_n_steps = steps_between_vals
print(f"Evaluating {round(steps_per_epoch / eval_every_n_steps, 1)} times per epoch", flush=True)

tracker = CategoricalLossTracker(sequence_types)
train_tracker = CategoricalLossTracker(sequence_types)

# All parameters trained at the same rate — no staged unfreezing needed
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_scratch)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.7,
    patience=6,
    min_lr=1e-7,
)

initial_lr = optimizer.param_groups[0]["lr"]
best_val_loss = float("inf")
threshold_patience = 16
counter_patience = 0
step = 0
val_times_counter = 0

scaler = GradScaler() if "cuda" in device_type else None
if scaler is not None:
    print("Mixed precision training enabled", flush=True)

train_losses = []
last_val_step = step

for epoch in range(epochs):
    model.train()

    for i, batch in enumerate(train_loader):
        train_losses, last_batch_loss = training_iteration(
            i, batch, scaler, model, optimizer, device, train_losses, train_tracker
        )

        if step - last_val_step >= eval_every_n_steps:
            print("Eval starting...", flush=True)
            clear_memory()
            model.eval()

            val_losses = []
            all_val_true_sequences = []
            all_val_pred_sequences = []
            all_val_sequence_types = []

            with torch.no_grad():
                tracker = CategoricalLossTracker(sequence_types)

                if torch.cuda.is_available():
                    print(f"[MEM] Val start | allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB", flush=True)

                for counter, val_batch in enumerate(val_loader):
                    v_inputs_nt_rf0 = val_batch["nt_encodings_rf0"].to(device, non_blocking=True)
                    v_inputs_nt_rf1 = val_batch["nt_encodings_rf1"].to(device, non_blocking=True)
                    v_inputs_nt_rf2 = val_batch["nt_encodings_rf2"].to(device, non_blocking=True)
                    v_encoded_labels = val_batch["label_encodings"].to(device, non_blocking=True).long()

                    padding_mask = v_encoded_labels == -1
                    valid_mask = ~padding_mask

                    if scaler is not None:
                        with autocast("cuda"):
                            v_outputs = model(v_inputs_nt_rf0, v_inputs_nt_rf1, v_inputs_nt_rf2, v_encoded_labels)
                            v_loss = v_outputs["loss"]
                    else:
                        v_outputs = model(v_inputs_nt_rf0, v_inputs_nt_rf1, v_inputs_nt_rf2, v_encoded_labels)
                        v_loss = v_outputs["loss"]

                    val_losses.append(v_loss.item())

                    if args.compute_sequence_metrics:
                        predictions = model.CRF.crf.decode(v_outputs["logits"], mask=valid_mask)
                        for seq_i, preds in enumerate(predictions):
                            true_seq = v_encoded_labels[seq_i][valid_mask[seq_i]].cpu().numpy()
                            all_val_true_sequences.append(true_seq)
                            all_val_pred_sequences.append(np.array(preds))
                        all_val_sequence_types.extend(list(val_batch["seq_desc"]))

                    seq_descs_batch = val_batch["seq_desc"]

                    for desc in tracker.categories:
                        desc_mask = torch.tensor([d == desc for d in seq_descs_batch], device=device)
                        if desc_mask.any():
                            desc_count = desc_mask.sum().item()
                            desc_logits = v_outputs["logits"][desc_mask]
                            desc_labels_original = v_encoded_labels[desc_mask]
                            desc_valid_mask = valid_mask[desc_mask]
                            safe_desc_labels = desc_labels_original.clone()
                            safe_desc_labels[safe_desc_labels == -1] = 0
                            desc_ll = model.CRF.crf(desc_logits, safe_desc_labels, mask=desc_valid_mask, reduction="none")
                            tracker.update(desc, (-desc_ll.mean()).item(), desc_count)
                            del desc_logits, desc_labels_original, safe_desc_labels, desc_valid_mask, desc_ll
                        del desc_mask

                    if counter % 30 == 0:
                        del v_inputs_nt_rf0, v_inputs_nt_rf1, v_inputs_nt_rf2, v_outputs, v_loss
                        clear_memory()

            val_avg_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")
            sequence_metrics = (
                calculate_sequence_accuracy_metrics(all_val_true_sequences, all_val_pred_sequences, all_val_sequence_types)
                if args.compute_sequence_metrics and all_val_true_sequences and all_val_pred_sequences
                else {}
            )
            train_avg_loss = sum(train_losses) / len(train_losses) if train_losses else float("inf")

            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                torch.save(
                    model.state_dict(),
                    f"{models_output_dir_path}/codon_only_model_{dataset_size}_d{args.d_model}_seed_{args.seed}{model_checkpoint_extension}.pth",
                )
                counter_patience = 0
            else:
                counter_patience += 1

            if counter_patience >= threshold_patience:
                print("Early stopping triggered!", flush=True)
                break

            scheduler.step(val_avg_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            lr_percentage = (current_lr / initial_lr) * 100

            val_times_counter, wandb_metrics = log_evaluation_metrics(
                epoch,
                train_avg_loss,
                val_avg_loss,
                best_val_loss,
                tracker,
                sequence_metrics,
                val_times_counter,
                sequence_types,
                train_tracker,
            )

            all_metrics = {**wandb_metrics, "lr_percent_of_initial": lr_percentage}
            if last_batch_loss is not None:
                all_metrics["train_loss_at_val_checkpoint"] = last_batch_loss

            wandb.log(all_metrics)

            del val_losses, all_val_true_sequences, all_val_pred_sequences, all_val_sequence_types
            clear_memory()

            train_losses = []
            last_val_step = step
            train_tracker = CategoricalLossTracker(sequence_types)
            model.train()

        step += 1

    if counter_patience >= threshold_patience:
        break

wandb.finish()
