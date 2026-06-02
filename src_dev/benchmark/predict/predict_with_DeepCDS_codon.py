"""
Codon-Only DeepCDS Prediction Script

Runs CDS predictions using the codon-only (nucleotide one-hot) DeepCDS model.
No ESM-2 / protein language model is used.
"""

import argparse
import gc
import gzip
import logging
import math
import os
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src_dev.modeling import TRAINED_WINDOW_SIZE_AA, get_window_positions

logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

torch.cuda.empty_cache()
pd.options.mode.chained_assignment = None
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

################################################################################################################################
################################################Argument Parser#################################################################
################################################################################################################################

parser = argparse.ArgumentParser(description="Run codon-only DeepCDS predictions")
parser.add_argument("--gpu", type=int, default=0, help="GPU number to use (default: 0)")
parser.add_argument("--healthtech_cluster", action="store_true", help="Whether running on HealthTech cluster")
parser.add_argument("--scarb_cluster", action="store_true", help="Whether running on SCARB cluster")
parser.add_argument(
    "--model",
    type=str,
    default="all_genomes",
    choices=["100_genomes", "200_genomes", "400_genomes", "all_genomes"],
    help="Model variant to load (default: all_genomes)",
)
parser.add_argument(
    "--error_type",
    type=str,
    default="substitution",
    choices=["indel_substitution", "substitution", "none"],
    help="Type of data errors the model was trained on (default: substitution)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="Batch size for inference (default: 256)",
)
parser.add_argument(
    "--stride_aa",
    type=int,
    default=70,
    help="Sliding window stride in codons for long sequences (default: 70, overlap=30 codons)",
)
parser.add_argument(
    "--input_format",
    type=str,
    default="csv",
    choices=["csv", "fasta"],
    help="Input file format: 'csv' (default) or 'fasta'",
)
parser.add_argument(
    "--d_model",
    type=int,
    default=332,
    help="Hidden dimension used during training (must match checkpoint; default: 332)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed used during training (must match checkpoint; default: 42)",
)

args = parser.parse_args()

# Set variables based on error type
if args.error_type == "indel_substitution":
    model_dir_path_suffix = "model_with_errors"
    label_classes = 6
elif args.error_type == "substitution":
    model_dir_path_suffix = "model_with_substitution_errors"
    label_classes = 4
else:
    model_dir_path_suffix = "model_without_errors"
    label_classes = 4

################################################################################################################################
################################################Device and Path Configuration###################################################
################################################################################################################################

if args.healthtech_cluster:
    base_data_path = "/net/well/pool/projects2/lisani/DeepCDS/FragmentPredictor/data/processed_data"
    input_data_dir_path = f"{base_data_path}/model_data/shared_crf/{model_dir_path_suffix}"
    num_workers_cpu = 2
    pin_memory = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type
    assert device_type == "cuda", "HealthTech cluster run should be on a CUDA GPU."

elif args.scarb_cluster:
    base_data_path = "/tmp/nrt204/FragmentPredictor/data/processed_data"
    input_data_dir_path = f"{base_data_path}/model_data/shared_crf/{model_dir_path_suffix}"
    num_workers_cpu = 2
    pin_memory = True
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    device_type = device.type
    assert device_type == "cuda", "SCARB cluster run should be on a CUDA GPU."

else:
    base_data_path = "../../../data/processed_data"
    input_data_dir_path = f"{base_data_path}/model_data/shared_crf/{model_dir_path_suffix}"
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device_type = device.type
    num_workers_cpu = 4 if device_type == "cuda" else 0
    pin_memory = device_type == "cuda"

print(f"Device: {device}", flush=True)
print(f"DataLoader workers: {num_workers_cpu}, pin_memory: {pin_memory}")

model_name_ckpt = f"codon_only_model_{args.model}_d{args.d_model}_seed_{args.seed}_final.pth"

test_samples_file = open(f"{base_data_path}/genome_partitions/test_partition_accessions.txt", "r")
test_samples = [line.strip() for line in test_samples_file.readlines()]
test_samples_file.close()


################################################################################################################################
################################################Codon-Only Model Architecture###################################################
################################################################################################################################

class PositionalEncoding(nn.Module):
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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class CodonTransformerEncoderBlock(nn.Module):
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
        x = self.input_projection(encoded_seqs_nt)
        x = self.pos_encoding(x)
        x = x.permute(1, 0, 2)
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        x = x.permute(1, 0, 2)
        x = self.norm(x)
        return self.linear(x)


class LinearChainCRF(nn.Module):
    def __init__(self, mapping_dict_to_class, transition_weight, num_encoded_labels=None, label_classes=4):
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
                0: {0, 2}, 1: {1, 3}, 2: {1}, 3: {0, 2},
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
        for from_label in range(num_labels):
            for to_label in range(num_labels):
                from_rf = self.shared_rf_labels_mapping[from_label]
                to_rf = self.shared_rf_labels_mapping[to_label]
                self.biologically_valid_mask[from_label, to_label] = self._is_legal_transition(from_rf, to_rf)

    def _create_frequent_transition_mask(self):
        frequent_rf_combinations = {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)}
        frequent_labels = [
            label_idx
            for label_idx, rf_combo in self.shared_rf_labels_mapping.items()
            if rf_combo in frequent_rf_combinations
        ]
        for label in frequent_labels:
            self.frequent_transition_mask[label, label] = True

    def forward(self, logits, attention_mask, labels=None):
        if labels is not None:
            crf_mask = labels != -1
            safe_labels = labels.clone()
            safe_labels[safe_labels == -1] = 0
            log_likelihood = self.crf(logits, safe_labels, mask=crf_mask, reduction="none")
            return {"loss": -log_likelihood.mean(), "logits": logits}
        else:
            crf_mask = attention_mask.bool()
            predictions = self.crf.decode(logits, mask=crf_mask)
            return {"predictions": predictions, "logits": logits}


class CodonOnlyCDSPredictor(nn.Module):
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
        label_classes=4,
        max_len=100,
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

    def forward(self, encoded_seqs_nt_rf0, encoded_seqs_nt_rf1, encoded_seqs_nt_rf2, labels=None):
        attention_mask = encoded_seqs_nt_rf0.sum(dim=-1) != 0
        logits_rf0 = self.CodonTransformerEncoderBlock(encoded_seqs_nt_rf0, attention_mask)
        logits_rf1 = self.CodonTransformerEncoderBlock(encoded_seqs_nt_rf1, attention_mask)
        logits_rf2 = self.CodonTransformerEncoderBlock(encoded_seqs_nt_rf2, attention_mask)
        combined = torch.cat([logits_rf0, logits_rf1, logits_rf2], dim=-1)
        logits_encoded = self.linear_transform(self.pre_crf_norm(combined))
        return self.CRF(logits=logits_encoded, attention_mask=attention_mask, labels=labels)


################################################################################################################################
################################################Model Loading###################################################################
################################################################################################################################

def load_codon_model(model_name_ckpt, input_data_dir_path, device, d_model, label_classes):
    with open(f"{input_data_dir_path}/label_mappings/mapping_to_3d_vector.pkl", "rb") as f:
        mapping_dict_to_class = pickle.load(f)

    num_encoded_labels = len(mapping_dict_to_class)
    print(f"Number of encoded label classes: {num_encoded_labels}")

    cfg = OmegaConf.load(f"{input_data_dir_path}/hyperparameter_configs/full_model_hyperparameters.yaml")

    act_function = cfg.hyperparameters.act_function
    num_layers = cfg.hyperparameters.depth_transformer_encoder_blocks
    n_attention_heads = cfg.hyperparameters.n_attention_heads
    dropout_rate = cfg.hyperparameters.dropout_rate_2
    transition_weight = cfg.hyperparameters.transition_weight

    assert d_model % n_attention_heads == 0, (
        f"d_model ({d_model}) must be divisible by n_attention_heads ({n_attention_heads})."
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
        max_len=TRAINED_WINDOW_SIZE_AA,
    )
    model.to(device)

    checkpoint = torch.load(
        f"{input_data_dir_path}/models/{model_name_ckpt}", map_location=device
    )
    model.load_state_dict(checkpoint, strict=True)
    print(f"Successfully loaded model from {model_name_ckpt}")

    return model, mapping_dict_to_class


################################################################################################################################
################################################Data Encoding###################################################################
################################################################################################################################

def _one_hot_encode(sequence):
    seq_len = len(sequence)
    encoding = torch.zeros(4, seq_len, dtype=torch.float32)
    seq_bytes = torch.frombuffer(bytearray(sequence.encode("ascii")), dtype=torch.uint8)
    for nuc, idx in ((65, 0), (67, 1), (71, 2), (84, 3)):
        encoding[idx, seq_bytes == nuc] = 1.0
    return encoding


def _nt_to_codon_tensor(sequences, max_aa_len):
    stacked = torch.stack(sequences)
    batch_size = stacked.shape[0]
    reshaped = stacked.view(batch_size, 4, max_aa_len, 3)
    transposed = reshaped.permute(0, 2, 1, 3)
    return transposed.reshape(batch_size, max_aa_len, 12)


class SeqDatasetCodon(Dataset):
    """Inference-only dataset for the codon-only model (no labels)."""

    def __init__(self, nt_encodings_rf0, nt_encodings_rf1, nt_encodings_rf2,
                 seq_errors, cds_coords, read_name):
        self.nt_encodings_rf0 = nt_encodings_rf0
        self.nt_encodings_rf1 = nt_encodings_rf1
        self.nt_encodings_rf2 = nt_encodings_rf2
        self.seq_errors = seq_errors
        self.cds_coords = cds_coords
        self.read_name = read_name

    def __getitem__(self, idx):
        return {
            "nt_encodings_rf0": torch.as_tensor(self.nt_encodings_rf0[idx], dtype=torch.float32),
            "nt_encodings_rf1": torch.as_tensor(self.nt_encodings_rf1[idx], dtype=torch.float32),
            "nt_encodings_rf2": torch.as_tensor(self.nt_encodings_rf2[idx], dtype=torch.float32),
            "seq_errors": str(self.seq_errors[idx]),
            "cds_coords": self.cds_coords[idx],
            "read_name": self.read_name[idx],
        }

    def __len__(self):
        return len(self.read_name)


def encode_data_codon(df, max_aa_len):
    """
    Encode raw reads into codon one-hot tensors for the three reading frames.

    Args:
        df (DataFrame): Must have columns 'read', 'cds_coords', 'indel_positions', 'read_name'.
        max_aa_len (int): Maximum number of codons per window.

    Returns:
        SeqDatasetCodon
    """
    max_nt_len = max_aa_len * 3
    encodings = {}

    for rf_idx, rf in enumerate(["rf0", "rf1", "rf2"]):
        # Extract RF-specific nucleotide subsequence
        nt_seqs = df["read"].apply(lambda seq: seq[rf_idx:])

        # Pad short sequences with N (→ all-zero codon = padding)
        nt_seqs = nt_seqs.apply(
            lambda seq: seq + "N" * (max_nt_len - len(seq)) if len(seq) < max_nt_len else seq[:max_nt_len]
        )

        one_hot_seqs = [_one_hot_encode(seq) for seq in nt_seqs]
        codon_tensors = _nt_to_codon_tensor(one_hot_seqs, max_aa_len)
        encodings[rf] = codon_tensors.numpy()

    return SeqDatasetCodon(
        encodings["rf0"], encodings["rf1"], encodings["rf2"],
        df["indel_positions"].tolist(),
        df["cds_coords"].tolist(),
        df["read_name"].tolist(),
    )


def parse_fasta_gz_to_df(fasta_gz_path):
    rows = []
    with gzip.open(fasta_gz_path, "rt") as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    parts = header.split("|")
                    rows.append({
                        "read_name": parts[0],
                        "read": "".join(seq_lines),
                        "cds_coords": parts[3] if len(parts) > 3 else "[]",
                        "indel_positions": parts[4] if len(parts) > 4 else "None",
                    })
                header = line[1:]
                seq_lines = []
            elif line:
                seq_lines.append(line)
        if header is not None:
            parts = header.split("|")
            rows.append({
                "read_name": parts[0],
                "read": "".join(seq_lines),
                "cds_coords": parts[3] if len(parts) > 3 else "[]",
                "indel_positions": parts[4] if len(parts) > 4 else "None",
            })
    return pd.DataFrame(rows)


def load_and_process_data(test_sample, data_dir, batch_size, max_aa_len,
                          num_workers_cpu=num_workers_cpu, pin_memory=pin_memory):
    if args.input_format == "fasta":
        test_set = parse_fasta_gz_to_df(
            f"{base_data_path}/reads_processed/test/{data_dir}/fasta/{test_sample}.fasta.gz"
        )
    else:
        test_set = pd.read_csv(
            f"{base_data_path}/reads_processed/test/{data_dir}/csv/{test_sample}.csv.gz",
            index_col=None,
            compression="gzip",
        )

    print("Data samples: ", test_set.shape[0])
    test_data = encode_data_codon(test_set, max_aa_len)
    del test_set

    return DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_cpu,
        pin_memory=pin_memory,
    )


################################################################################################################################
################################################Helper Functions################################################################
################################################################################################################################

def clear_memory(sync=False):
    if torch.cuda.is_available():
        if sync:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def trim_predictions_by_length(preds_rf0, preds_rf1, preds_rf2, seq_len):
    """Trim per-RF predictions to the actual number of codons for each frame."""
    rf0_len = seq_len // 3
    rf1_len = (seq_len - 1) // 3
    rf2_len = (seq_len - 2) // 3
    return (
        [p[:rf0_len] for p in preds_rf0],
        [p[:rf1_len] for p in preds_rf1],
        [p[:rf2_len] for p in preds_rf2],
    )


################################################################################################################################
################################################CDS Coordinate Extraction#######################################################
################################################################################################################################

@dataclass
class CDSSegment:
    start: int
    end: int
    frame: int
    start_type: str
    end_type: str
    group_id: Optional[str] = None
    indel_type: Optional[str] = None


@dataclass
class Transition:
    type: str
    start_position: int
    end_position: int
    frame: int


@dataclass
class UncertainRegion:
    start: int
    end: int
    overlapping_frames: List[int]
    reason: str


def get_cds_coords(labels_rf0, labels_rf1, labels_rf2):
    uncertain_regions = []
    transition_positions = {
        "start_codon": [], "stop_codon": [], "indel_start": [], "indel_stop": []
    }
    all_cds_fragments = []
    transitions_info = []

    for rf, labels in enumerate([labels_rf0, labels_rf1, labels_rf2]):
        labels = np.array(labels)
        frame_segments, start_stop_codon_transitions = _extract_segments_from_frame(
            labels, rf, transition_positions
        )
        all_cds_fragments.extend(frame_segments)
        transitions_info.extend(start_stop_codon_transitions)

    all_cds_fragments.sort(key=lambda x: x.start)
    connected_segments = _connect_frameshift_segments(all_cds_fragments)
    uncertain_regions, transitions_info = _create_uncertain_regions_from_groups(
        connected_segments, transitions_info
    )
    connected_segments.sort(key=lambda x: x.start)
    transitions_info.sort(key=lambda x: x.start_position)

    return connected_segments, uncertain_regions, transitions_info, transition_positions


def _extract_segments_from_frame(labels, rf, transition_positions):
    segments = []
    start_stop_codon_transitions = []
    in_cds = False
    start = None
    start_type = None

    for i, label in enumerate(labels):
        nt_pos = i * 3 + rf + 1

        if label in [1, 2, 4]:
            if not in_cds:
                in_cds = True
                start = nt_pos
                if label == 2:
                    start_type = "start_codon"
                    transition_positions["start_codon"].append(nt_pos)
                    start_stop_codon_transitions.append(
                        Transition(type="start_codon", start_position=nt_pos, end_position=nt_pos + 2, frame=rf)
                    )
                elif label == 4:
                    start_type = "indel_start"
                    transition_positions["indel_start"].append(nt_pos)
                else:
                    start_type = "internal_region"

        elif label in [3, 5, 0]:
            if in_cds:
                if label == 3:
                    end_type = "stop_codon"
                    end = nt_pos + 2
                    transition_positions["stop_codon"].append(end)
                    start_stop_codon_transitions.append(
                        Transition(type="stop_codon", start_position=nt_pos, end_position=end, frame=rf)
                    )
                elif label == 5:
                    end_type = "indel_stop"
                    end = nt_pos + 2
                    transition_positions["indel_stop"].append(end)
                else:
                    end_type = "internal_region"
                    end = nt_pos - 1

                segments.append(CDSSegment(start=start, end=end, frame=rf,
                                           start_type=start_type, end_type=end_type))
                in_cds = False
                start = None
                start_type = None

    if in_cds:
        end = len(labels) * 3 + rf
        segments.append(CDSSegment(start=start, end=end, frame=rf,
                                   start_type=start_type, end_type="internal_region"))

    return segments, start_stop_codon_transitions


def _create_uncertain_regions_from_groups(segments, transitions):
    uncertain_regions = []
    groups = defaultdict(list)
    for segment in segments:
        if segment.group_id:
            groups[segment.group_id].append(segment)

    for group_id, group_segments in groups.items():
        if len(group_segments) < 2:
            continue
        group_segments.sort(key=lambda x: x.start)

        for i in range(len(group_segments) - 1):
            seg1 = group_segments[i]
            seg2 = group_segments[i + 1]

            if seg1.end >= seg2.start:
                overlap_start = seg2.start
                positions_before_overlap = overlap_start - seg1.start
                complete_codons_in_seg1 = positions_before_overlap // 3
                seg1_trim_end = seg1.start + (complete_codons_in_seg1 * 3) - 1

                overlap_end = seg1.end
                positions_in_overlap = overlap_end - seg2.start + 1
                codons_to_skip = (positions_in_overlap + 2) // 3
                seg2_trim_start = seg2.start + (codons_to_skip * 3)

                if seg1_trim_end >= seg1.start and seg2_trim_start <= seg2.end:
                    seg1.end = seg1_trim_end
                    seg2.start = seg2_trim_start

                    uncertain_start = seg1.end + 1
                    uncertain_end = seg2.start - 1

                    if uncertain_end > uncertain_start:
                        uncertain_regions.append(UncertainRegion(
                            start=uncertain_start, end=uncertain_end,
                            overlapping_frames=[seg1.frame, seg2.frame],
                            reason=f"Frameshift overlap between RF{seg1.frame} and RF{seg2.frame}",
                        ))
                    elif uncertain_end == uncertain_start:
                        transitions.append(Transition(
                            type="insertion", start_position=uncertain_start,
                            end_position=uncertain_end, frame=seg1.frame,
                        ))
            else:
                gap_start = seg1.end + 1
                gap_end = seg2.start - 1
                if gap_end > gap_start:
                    uncertain_regions.append(UncertainRegion(
                        start=gap_start, end=gap_end,
                        overlapping_frames=[seg1.frame, seg2.frame],
                        reason=f"Frameshift gap between RF{seg1.frame} and RF{seg2.frame}",
                    ))
                elif gap_end == gap_start:
                    transitions.append(Transition(
                        type="insertion", start_position=gap_start,
                        end_position=gap_end, frame=seg1.frame,
                    ))

    return uncertain_regions, transitions


def detect_indel_type(from_frame, to_frame):
    if from_frame == to_frame:
        return None
    forward_jumps = {(0, 1), (1, 2), (2, 0)}
    backward_jumps = {(0, 2), (1, 0), (2, 1)}
    transition = (from_frame, to_frame)
    if transition in forward_jumps:
        return "insertion"
    elif transition in backward_jumps:
        return "deletion"
    return "complex"


def _connect_frameshift_segments(segments):
    connected_segments = []
    used_segments = set()
    group_counter = 1

    for i, segment in enumerate(segments):
        if i in used_segments:
            continue

        current_group = [segment]
        used_segments.add(i)

        if segment.end_type == "indel_stop":
            for j, other_segment in enumerate(segments[i + 1:], i + 1):
                if (j not in used_segments and
                        other_segment.start_type == "indel_start" and
                        other_segment.frame != segment.frame and
                        abs(other_segment.start - segment.end) <= 30):
                    indel_type = detect_indel_type(segment.frame, other_segment.frame)
                    segment.indel_type = indel_type
                    other_segment.indel_type = indel_type
                    current_group.append(other_segment)
                    used_segments.add(j)

                    last_segment = other_segment
                    for k, next_segment in enumerate(segments[j + 1:], j + 1):
                        if (k not in used_segments and
                                last_segment.end_type == "indel_stop" and
                                next_segment.start_type == "indel_start" and
                                next_segment.frame != last_segment.frame and
                                abs(next_segment.start - last_segment.end) <= 30):
                            next_indel_type = detect_indel_type(last_segment.frame, next_segment.frame)
                            last_segment.indel_type = next_indel_type
                            next_segment.indel_type = next_indel_type
                            current_group.append(next_segment)
                            used_segments.add(k)
                            last_segment = next_segment
                        else:
                            break
                    break

        if len(current_group) > 1:
            group_id = f"group_{group_counter}"
            for seg in current_group:
                seg.group_id = group_id
            group_counter += 1

        connected_segments.extend(current_group)

    return connected_segments


def write_enhanced_gff(segments, uncertain_regions, transitions_info, read_name, cds_coords, seq_errors, outfile_gff):
    counter_cds_frags_interrupted = {}

    for i, segment in enumerate(segments):
        attributes = [f"start={segment.start_type}", f"end={segment.end_type}"]

        if segment.group_id:
            if segment.group_id not in counter_cds_frags_interrupted:
                counter_cds_frags_interrupted[segment.group_id] = 0
            else:
                counter_cds_frags_interrupted[segment.group_id] += 1
            attributes.append(f"group_id={segment.group_id}.{counter_cds_frags_interrupted[segment.group_id]}")

        if segment.indel_type:
            attributes.append(f"indel_type={segment.indel_type}")

        attributes.append(f"ref={cds_coords}")
        attributes.append(f"seq_errors={seq_errors}")

        outfile_gff.write(
            f"{read_name}\tCodonPredictor\tCDS\t{segment.start}\t{segment.end}\t"
            f".\t+\t{segment.frame}\t{';'.join(attributes)}\n"
        )

    for i, transition in enumerate(transitions_info):
        outfile_gff.write(
            f"{read_name}\tCodonPredictor\t{transition.type}\t{transition.start_position}\t{transition.end_position}\t"
            f".\t+\t.\tID={transition.type}_{read_name}_{i}\n"
        )

    for i, region in enumerate(uncertain_regions):
        attributes = [
            f"Note=Uncertain region: {region.reason}",
            f"overlapping_frames={','.join(map(str, region.overlapping_frames))}",
        ]
        involved_groups = {
            seg.group_id for seg in segments
            if seg.group_id and not (seg.end < region.start or seg.start > region.end)
        }
        if involved_groups:
            attributes.append(f"involved_groups={','.join(involved_groups)}")
        outfile_gff.write(
            f"{read_name}\tCodonPredictor\tuncertain_region\t{region.start}\t{region.end}\t"
            f".\t+\t.\t{';'.join(attributes)}\n"
        )


def process_predictions_enhanced(predictions_rf0, predictions_rf1, predictions_rf2,
                                  read_names, cds_coords, seq_errors, outfile_gff, batch_size):
    for i in range(min(batch_size, len(cds_coords))):
        segments, uncertain_regions, transitions_info, _ = get_cds_coords(
            predictions_rf0[i], predictions_rf1[i], predictions_rf2[i]
        )
        write_enhanced_gff(
            segments, uncertain_regions, transitions_info,
            read_names[i], cds_coords[i], seq_errors[i], outfile_gff,
        )


################################################################################################################################
################################################Main Prediction Loop############################################################
################################################################################################################################

def run_model_predictions(data_dir, model, mapping_dict_to_class, max_aa_len, seq_len,
                          test_samples=test_samples, batch_size=256, use_half_precision=True):
    if use_half_precision and device_type in ("cuda", "mps"):
        model = model.half()
        dtype = torch.float16
        print(f"Using half precision (FP16) inference on {device_type}")
    else:
        dtype = torch.float32

    for test_sample in tqdm(test_samples, desc="Processing samples"):
        test_loader = load_and_process_data(
            test_sample, data_dir=data_dir, batch_size=batch_size, max_aa_len=max_aa_len,
        )

        dir_path = (
            f"{base_data_path}/predictions/raw_predictions/DeepCDS_codon_only/"
            f"{model_dir_path_suffix}/{data_dir}/{model_name_ckpt.split('.')[0]}/"
        )
        os.makedirs(dir_path, exist_ok=True)
        outfile_gff = open(f"{dir_path}/predictions_{test_sample}.gff", "w")
        outfile_gff.write("##gff-version 3\n")

        with torch.inference_mode():
            model.eval()
            for counter, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
                nt_rf0 = batch["nt_encodings_rf0"].to(device, dtype=dtype)
                nt_rf1 = batch["nt_encodings_rf1"].to(device, dtype=dtype)
                nt_rf2 = batch["nt_encodings_rf2"].to(device, dtype=dtype)

                cds_coords = batch["cds_coords"]
                read_names = batch["read_name"]
                seq_errors = batch["seq_errors"]

                outputs = model(nt_rf0, nt_rf1, nt_rf2)
                predictions_encoded = outputs["predictions"]

                preds_rf0, preds_rf1, preds_rf2 = [], [], []
                for preds_sample in predictions_encoded:
                    decoded = [mapping_dict_to_class[p] for p in preds_sample]
                    preds_rf0.append([rf[0] for rf in decoded])
                    preds_rf1.append([rf[1] for rf in decoded])
                    preds_rf2.append([rf[2] for rf in decoded])

                preds_rf0, preds_rf1, preds_rf2 = trim_predictions_by_length(
                    preds_rf0, preds_rf1, preds_rf2, seq_len
                )

                process_predictions_enhanced(
                    preds_rf0, preds_rf1, preds_rf2,
                    read_names, cds_coords, seq_errors, outfile_gff, batch_size,
                )

                del nt_rf0, nt_rf1, nt_rf2, outputs, predictions_encoded
                del preds_rf0, preds_rf1, preds_rf2

                if (counter + 1) % 50 == 0:
                    clear_memory()

        outfile_gff.close()
        del test_loader
        clear_memory(sync=True)

    if use_half_precision and device_type in ("cuda", "mps"):
        model = model.float()


def _run_codon_model_on_windows(model, window_dataset, device, dtype, batch_size,
                                 num_workers_cpu=0, pin_memory=False):
    """Run codon-only model on windowed sequences and collect pre-CRF logits."""
    window_loader = DataLoader(
        window_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers_cpu, pin_memory=pin_memory,
    )
    all_logits = []
    for batch in window_loader:
        nt_rf0 = batch["nt_encodings_rf0"].to(device, dtype=dtype)
        nt_rf1 = batch["nt_encodings_rf1"].to(device, dtype=dtype)
        nt_rf2 = batch["nt_encodings_rf2"].to(device, dtype=dtype)
        outputs = model(nt_rf0, nt_rf1, nt_rf2)
        all_logits.append(outputs["logits"].float().detach())
        del nt_rf0, nt_rf1, nt_rf2, outputs
    return torch.cat(all_logits, dim=0)


def _merge_window_logits(window_logits, window_starts, window_size_aa, full_aa_len, num_labels, device):
    """Average logits from overlapping windows."""
    n_sequences = window_logits.shape[0]
    merged_logits = torch.zeros(n_sequences, full_aa_len, num_labels, dtype=torch.float32, device=device)
    overlap_count = torch.zeros(n_sequences, full_aa_len, 1, dtype=torch.float32, device=device)

    for w_idx, start_nt in enumerate(window_starts):
        start_aa = start_nt // 3
        actual_len = min(window_size_aa, full_aa_len - start_aa)
        merged_logits[:, start_aa:start_aa + actual_len, :] += window_logits[:, w_idx, :actual_len, :]
        overlap_count[:, start_aa:start_aa + actual_len, :] += 1

    merged_logits = merged_logits / overlap_count.clamp(min=1)
    merged_mask = (overlap_count.squeeze(-1) > 0)
    return merged_logits, merged_mask


def run_sliding_window_predictions(data_dir, model, mapping_dict_to_class, seq_len,
                                    test_samples=test_samples, batch_size=256,
                                    stride_aa=70, use_half_precision=True):
    if use_half_precision and device_type in ("cuda", "mps"):
        model = model.half()
        dtype = torch.float16
        print(f"Using half precision (FP16) inference on {device_type}")
    else:
        dtype = torch.float32

    window_size_aa = TRAINED_WINDOW_SIZE_AA
    window_size_nt = window_size_aa * 3
    stride_nt = stride_aa * 3

    window_starts = get_window_positions(seq_len, window_size_nt, stride_nt)
    n_windows = len(window_starts)
    full_aa_len = seq_len // 3
    num_labels = model.linear_transform.out_features
    effective_batch_size = max(1, batch_size // n_windows)

    print(f"  Sliding window: {n_windows} windows per sequence "
          f"(size={window_size_nt}nt, stride={stride_nt}nt, overlap={window_size_nt - stride_nt}nt)")
    print(f"  Effective batch size: {effective_batch_size} sequences")

    for test_sample in tqdm(test_samples, desc="Processing samples"):
        if args.input_format == "fasta":
            test_df = parse_fasta_gz_to_df(
                f"{base_data_path}/reads_processed/test/{data_dir}/fasta/{test_sample}.fasta.gz"
            )
        else:
            test_df = pd.read_csv(
                f"{base_data_path}/reads_processed/test/{data_dir}/csv/{test_sample}.csv.gz",
                index_col=None, compression="gzip",
            )

        print(f"Data samples: {test_df.shape[0]}")

        dir_path = (
            f"{base_data_path}/predictions/raw_predictions/DeepCDS_codon_only/"
            f"{model_dir_path_suffix}/{data_dir}/{model_name_ckpt.split('.')[0]}/"
        )
        os.makedirs(dir_path, exist_ok=True)
        outfile_gff = open(f"{dir_path}/predictions_{test_sample}.gff", "w")
        outfile_gff.write("##gff-version 3\n")

        n_sequences = len(test_df)

        with torch.inference_mode():
            model.eval()
            chunk_counter = 0

            for chunk_start in range(0, n_sequences, effective_batch_size):
                chunk_df = test_df.iloc[chunk_start:chunk_start + effective_batch_size].reset_index(drop=True)
                chunk_size = len(chunk_df)

                # Expand each sequence into n_windows windowed rows
                rows = []
                for _, row in chunk_df.iterrows():
                    seq = row["read"]
                    for start in window_starts:
                        rows.append({
                            "read": seq[start:start + window_size_nt],
                            "cds_coords": row["cds_coords"],
                            "indel_positions": row["indel_positions"],
                            "read_name": row["read_name"],
                        })
                windowed_df = pd.DataFrame(rows)

                window_dataset = encode_data_codon(windowed_df, window_size_aa)
                total_windows = chunk_size * n_windows

                all_logits = _run_codon_model_on_windows(
                    model, window_dataset, device, dtype,
                    batch_size=batch_size,
                    num_workers_cpu=num_workers_cpu,
                    pin_memory=pin_memory,
                )

                # (total_windows, window_size_aa, L) → (chunk_size, n_windows, window_size_aa, L)
                all_logits = all_logits.view(chunk_size, n_windows, window_size_aa, num_labels)

                merged_logits, merged_mask = _merge_window_logits(
                    all_logits, window_starts, window_size_aa, full_aa_len, num_labels, device
                )

                if dtype == torch.float16:
                    merged_logits = merged_logits.half()
                merged_mask = merged_mask.bool()

                predictions_encoded = model.CRF.crf.decode(merged_logits, mask=merged_mask)

                preds_rf0, preds_rf1, preds_rf2 = [], [], []
                for preds_sample in predictions_encoded:
                    decoded = [mapping_dict_to_class[p] for p in preds_sample]
                    preds_rf0.append([rf[0] for rf in decoded])
                    preds_rf1.append([rf[1] for rf in decoded])
                    preds_rf2.append([rf[2] for rf in decoded])

                preds_rf0, preds_rf1, preds_rf2 = trim_predictions_by_length(
                    preds_rf0, preds_rf1, preds_rf2, seq_len
                )

                read_names = chunk_df["read_name"].tolist()
                cds_coords = chunk_df["cds_coords"].tolist()
                seq_errors = chunk_df["indel_positions"].astype(str).tolist()

                process_predictions_enhanced(
                    preds_rf0, preds_rf1, preds_rf2,
                    read_names, cds_coords, seq_errors, outfile_gff, chunk_size,
                )

                chunk_counter += 1
                if chunk_counter % 10 == 0:
                    clear_memory()

                del all_logits, merged_logits, merged_mask, windowed_df, window_dataset
                del predictions_encoded

        outfile_gff.close()
        del test_df
        clear_memory(sync=True)

    if use_half_precision and device_type in ("cuda", "mps"):
        model = model.float()


################################################################################################################################
################################################Main Entry Point################################################################
################################################################################################################################

if __name__ == "__main__":
    print(f"Model checkpoint: {model_name_ckpt}")
    print(f"Error type: {args.error_type}")
    print(f"Batch size: {args.batch_size}")
    print(f"d_model: {args.d_model}, seed: {args.seed}")

    if args.error_type == "none":
        if args.model == "all_genomes":
            data_dirs = [
                "without_errors_60bp",
                "without_errors_75bp",
                "without_errors_100bp",
                "without_errors_150bp",
                "without_errors_300bp",
                "without_errors_700bp",
                "without_errors_1000bp",
            ]
        else:
            data_dirs = ["without_errors_300bp"]
            print(f"Note: Using only 300bp dataset for model '{args.model}' (use --model all_genomes for full evaluation)")

    elif args.error_type in ("indel_substitution", "substitution"):
        error_profiles = [
            "with_errors_5e-06i_0.004s",
            #"with_errors_1.25e-05i_0.01s",
            #"with_errors_3.75e-05i_0.03s",
        ]
        if args.model == "all_genomes":
            read_lengths = ["300bp"] #["60bp", "75bp", "100bp", "150bp", "300bp"]
            data_dirs = [f"{profile}_{length}" for profile in error_profiles for length in read_lengths[::-1]]
            data_dirs += ["HiSeq2500_150bp", "MiSeq_v3_300bp", "NextSeq500_150bp"]
        else:
            data_dirs = [f"{profile}_300bp" for profile in error_profiles]
            print(f"Note: Using only 300bp datasets for model '{args.model}' (use --model all_genomes for full evaluation)")

    else:
        raise ValueError(f"Unknown error_type: '{args.error_type}'")

    model, mapping_dict_to_class = load_codon_model(
        model_name_ckpt,
        input_data_dir_path,
        device=device,
        d_model=args.d_model,
        label_classes=label_classes,
    )

    trained_window_nt = TRAINED_WINDOW_SIZE_AA * 3  # 300 nt

    for data_dir in data_dirs:
        print(data_dir, flush=True)
        seq_len = int(data_dir.split("_")[-1].strip("bp"))

        if seq_len > trained_window_nt:
            print(f"  Using sliding window inference (seq_len={seq_len} > trained window={trained_window_nt})")
            run_sliding_window_predictions(
                data_dir, model, mapping_dict_to_class, seq_len,
                batch_size=args.batch_size, stride_aa=args.stride_aa,
            )
        else:
            max_aa_len = min(int(np.ceil(seq_len / 3)) + 5, TRAINED_WINDOW_SIZE_AA)
            run_model_predictions(
                data_dir, model, mapping_dict_to_class, max_aa_len, seq_len,
                batch_size=args.batch_size,
            )
