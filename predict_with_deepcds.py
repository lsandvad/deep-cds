"""
DeepCDS Prediction Script — User-facing

Predict coding sequences (CDS) in nucleotide FASTA sequences using trained
DeepCDS models. Supports variable-length input sequences and three model
variants trained on different error profiles.

Usage examples:
    # No sequencing errors (clean sequences)
    python predict_with_deepcds.py --fasta input.fasta --error_type none --output predictions.gff

    # Sequences with substitution errors (e.g. Illumina)
    python predict_with_deepcds.py --fasta input.fasta --error_type substitution --output predictions.gff

    # Sequences with indel + substitution errors
    python predict_with_deepcds.py --fasta input.fasta --error_type indel_substitution --output predictions.gff
"""

import argparse
import gc
import logging
import os
import io
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional
from Bio import SeqIO
import csv

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src import TRAINED_WINDOW_SIZE_AA, encode_data, load_model
from src.sliding_window import (
    _create_windowed_dataframe,
    _decode_predictions,
    _merge_window_logits,
    _run_model_on_windows,
    get_window_positions,
)

# Suppress noisy third-party library warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message="enable_nested_tensor", category=UserWarning)

logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════════════════════
# Argument Parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict coding sequences (CDS) in nucleotide FASTA sequences using DeepCDS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_with_deepcds.py --fasta reads.fasta --error_type none
  python predict_with_deepcds.py --fasta reads.fasta --error_type substitution --output my_predictions.gff
  python predict_with_deepcds.py --fasta reads.fasta --error_type indel_substitution --batch_size 128
        """,
    )
    parser.add_argument(
        "--fasta",
        type=str,
        required=True,
        help="Path to input FASTA file with nucleotide sequences",
    )
    parser.add_argument(
        "--error_type",
        type=str,
        required=True,
        choices=["none", "substitution", "indel_substitution"],
        help=(
            "Error profile the model was trained on: "
            "'none' for error-free sequences, "
            "'substitution' for sequences with substitution errors, "
            "'indel_substitution' for sequences with indel and substitution errors"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path and name without file format extension (default: <fasta_stem>_deepcds_predictions)",
    )
    parser.add_argument('--compute_device',
                        type = str,
                        default = "cuda",
                        choices=["cuda", "mps", "cpu"], 
                        help='Hardware accelerator to use. Options: "cuda" (NVIDIA GPU), "mps" (Apple Silicon), or "cpu". The program will automatically fall back to CPU if the requested device is unavailable.')
        

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for inference (how many sequences are processed together in one iteration). If you have limited memory, try a smaller batch size (default: 256)",
    )

    parser.add_argument(
        "--min_cds_length",
        type=int,
        default=60,
        help="Minimum length for predicted CDS sequences. We recommend not going below 30 nt as this may affect prediction accuracy to a large extent (default: 60)",
    )

    parser.add_argument(
        "--stride_aa",
        type=int,
        default=50,
        help="Sliding window stride in codons for long sequences (how many codons the prediction window advances between each inference step). Smaller stride gives larger overlap between consecutive windows and may improve accuracy, but increases computation time (default: 50)",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# FASTA Parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_fasta(fasta_path):
    """
    Parse a FASTA file and return a list of (name, sequence) tuples.

    Handles multi-line sequences and strips whitespace. Sequence names are
    taken from the first word of the header line (after '>').
    """
    sequences = []
    current_name = None
    current_seq_parts = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    sequences.append((current_name, "".join(current_seq_parts).upper()))
                current_name = line[1:].split()[0]
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

    # Don't forget the last sequence
    if current_name is not None:
        sequences.append((current_name, "".join(current_seq_parts).upper()))

    return sequences


def validate_sequences(sequences):
    """
    Validate nucleotide sequences and warn about potential issues.

    Args:
        sequences: List of (name, sequence) tuples.
    """
    # Get sets of valid and certain nucleotides for quick checks
    valid_nucs = set("ACGTNRYSWKMBDHV")
    certain_nucs = set("ACGT")

    filtered = []
    for name, seq in sequences:
        #Replace U to T if present, as our model is trained on DNA sequences
        seq = seq.replace("U", "T")
        # Skip sequences shorter than 30 nt, a we only validate CDS fragments of >= 30 nt
        if len(seq) < 30:
            print(f"  Warning: Skipping '{name}' — sequence too short ({len(seq)} nt, minimum 30 nt)")
            continue
        invalid_chars = set(seq) - valid_nucs
        if invalid_chars:
            print(f"  Warning: '{name}' contains non-standard characters: {invalid_chars} — treating as N")

            #Convert all unknown/ambiguous chars to N
            seq = "".join(c if c in certain_nucs else "N" for c in seq)
        filtered.append((name, seq))
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def clear_memory(sync=False):
    """Memory clean up function."""
    if torch.cuda.is_available():
        if sync:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


_cached_tokenizer = None

def get_tokenizer():
    """Get cached ESM-2 tokenizer."""
    global _cached_tokenizer
    if _cached_tokenizer is None:
        _cached_tokenizer = AutoTokenizer.from_pretrained(
            "facebook/esm2_t6_8M_UR50D",
            do_lower_case=False,
        )
    return _cached_tokenizer


def sequences_to_dataframe(names, seqs):
    """Convert sequence names and sequences into a DataFrame matching encode_data's expected format."""
    return pd.DataFrame({
        "read": seqs,
        "read_name": names,
        "cds_coords": ["NA"] * len(names),
        "indel_positions": ["NA"] * len(names),
    })


def get_actual_sequence_length(input_ids, eos_token_id=2):
    """Find actual sequence length by locating EOS token."""
    actual_lengths = []
    for seq in input_ids:
        eos_positions = (seq == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            actual_length = eos_positions[0].item() - 1
        else:
            actual_length = len(seq) - 1
        actual_lengths.append(max(1, actual_length))
    return actual_lengths


def trim_predictions_by_eos(predictions, input_ids):
    """Trim predictions to actual sequence length based on EOS token."""
    actual_lengths = get_actual_sequence_length(input_ids, eos_token_id=2)
    trimmed = []
    for pred_seq, length in zip(predictions, actual_lengths):
        trimmed.append(pred_seq[:length])
    return trimmed


# ══════════════════════════════════════════════════════════════════════════════
# CDS Coordinate Extraction (reused from benchmark script)
# ══════════════════════════════════════════════════════════════════════════════

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
    """
    Get predicted CDS coordinates with frameshift handling if any. Returns connected CDS segments, uncertain regions, and transition info.

    Args: 
        - labels_rf0, labels_rf1, labels_rf2: Lists of predicted labels for each reading frame
    """

    # Initialize
    uncertain_regions = []
    transition_positions = {
        'start_codon': [], 'stop_codon': [],
        'indel_start': [], 'indel_stop': []}
    all_cds_fragments = []
    transitions_info = []

    # Extract CDS segments and transitions from each reading frame's predictions
    for rf, labels in enumerate([labels_rf0, labels_rf1, labels_rf2]):
        labels = np.array(labels)
        frame_segments, start_stop_transitions = _extract_segments_from_frame(labels, rf, transition_positions)
        all_cds_fragments.extend(frame_segments)
        transitions_info.extend(start_stop_transitions)

    # Connect frameshift segments, create uncertain regions, and sort final results
    all_cds_fragments.sort(key=lambda x: x.start)

    # Identify and connect frameshift segments that are predicted to belong to same CDS and locate potential uncertain regions between them
    connected_segments = _connect_frameshift_segments(all_cds_fragments)
    uncertain_regions, transitions_info = _create_uncertain_regions_from_groups(connected_segments, transitions_info)
    connected_segments.sort(key=lambda x: x.start)

    # Sort transitions (non-coding <-> CDS) by position for consistent output
    transitions_info.sort(key=lambda x: x.start_position)

    return connected_segments, uncertain_regions, transitions_info, transition_positions


def _extract_segments_from_frame(labels, rf, transition_positions):
    """Extract CDS segments from a single reading frame."""
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
                    start_type = 'start_codon'
                elif label == 4:
                    start_type = 'indel_start'
                    transition_positions['indel_start'].append(nt_pos)
                else:
                    start_type = 'internal_region'

        elif label in [3, 5, 0]:
            if in_cds:
                if label == 3:
                    end_type = 'stop_codon'
                    end = nt_pos + 2
                elif label == 5:
                    end_type = 'indel_stop'
                    end = nt_pos + 2
                    transition_positions['indel_stop'].append(end)
                else:
                    end_type = 'internal_region'
                    end = nt_pos - 1

                segments.append(CDSSegment(start=start, end=end, frame=rf,
                                           start_type=start_type, end_type=end_type))
                in_cds = False
                start = None
                start_type = None

    if in_cds:
        end = len(labels) * 3 + rf
        segments.append(CDSSegment(start=start, end=end, frame=rf,
                                   start_type=start_type, end_type='internal_region'))

    return segments, start_stop_codon_transitions


def detect_indel_type(from_frame, to_frame):
    """Detect indel type based on reading frame transition."""
    if from_frame == to_frame:
        return None
    forward_jumps = {(0, 1), (1, 2), (2, 0)}
    backward_jumps = {(0, 2), (1, 0), (2, 1)}
    transition = (from_frame, to_frame)
    if transition in forward_jumps:
        return 'insertion'
    elif transition in backward_jumps:
        return 'deletion'
    return 'complex'


def _connect_frameshift_segments(segments):
    """Connect segments that might be part of the same CDS interrupted by frameshifts."""
    connected_segments = []
    used_segments = set()
    group_counter = 1

    for i, segment in enumerate(segments):
        if i in used_segments:
            continue

        current_group = [segment]
        used_segments.add(i)

        if segment.end_type == 'indel_stop':
            for j, other_segment in enumerate(segments[i+1:], i+1):
                if (j not in used_segments and
                    other_segment.start_type == 'indel_start' and
                    other_segment.frame != segment.frame and
                    abs(other_segment.start - segment.end) <= 30):

                    indel_type = detect_indel_type(segment.frame, other_segment.frame)
                    segment.indel_type = indel_type
                    other_segment.indel_type = indel_type
                    current_group.append(other_segment)
                    used_segments.add(j)

                    last_segment = other_segment
                    for k, next_segment in enumerate(segments[j+1:], j+1):
                        if (k not in used_segments and
                            last_segment.end_type == 'indel_stop' and
                            next_segment.start_type == 'indel_start' and
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


def _create_uncertain_regions_from_groups(segments, transitions):
    """Create uncertain regions between connected frameshift segments."""
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
                            reason=f"Frameshift overlap between RF{seg1.frame} and RF{seg2.frame}"
                        ))
                    elif uncertain_end == uncertain_start:
                        transitions.append(Transition(
                            type="insertion", start_position=uncertain_start,
                            end_position=uncertain_end, frame=seg1.frame
                        ))
            else:
                gap_start = seg1.end + 1
                gap_end = seg2.start - 1

                if gap_end > gap_start:
                    uncertain_regions.append(UncertainRegion(
                        start=gap_start, end=gap_end,
                        overlapping_frames=[seg1.frame, seg2.frame],
                        reason=f"Frameshift gap between RF{seg1.frame} and RF{seg2.frame}"
                    ))
                elif gap_end == gap_start:
                    transitions.append(Transition(
                        type="insertion", start_position=gap_start,
                        end_position=gap_end, frame=seg1.frame
                    ))

    return uncertain_regions, transitions


# ══════════════════════════════════════════════════════════════════════════════
# GFF Output
# ══════════════════════════════════════════════════════════════════════════════

def write_gff(segments, uncertain_regions, transitions_info, read_name, outfile_gff, min_cds_length):
    """Write CDS predictions to GFF file."""
    counter_cds_frags_interrupted = {}

    for segment in segments:
        attributes = []
        attributes.append(f"start={segment.start_type}")
        attributes.append(f"end={segment.end_type}")

        if segment.group_id:
            if segment.group_id not in counter_cds_frags_interrupted:
                counter_cds_frags_interrupted[segment.group_id] = 0
            else:
                counter_cds_frags_interrupted[segment.group_id] += 1
            attributes.append(f"group_id={segment.group_id}.{counter_cds_frags_interrupted[segment.group_id]}")

        if segment.indel_type:
            attributes.append(f"indel_type={segment.indel_type}")

        # Discard complete CDS fragments and their start/stop codon annotations shorter than 30 bp. Only discard these if they are not interrupted by indels; TOGGLE LATER AS USER OPTION!!
        if segment.end - segment.start < min_cds_length and segment.indel_type == None:
            continue

        attr_string = ";".join(attributes)
        outfile_gff.write(
            f"{read_name}\tDeepCDS\tCDS\t{segment.start}\t{segment.end}\t"
            f".\t+\t{segment.frame}\t{attr_string}\n"
        )

        if segment.start_type == 'start_codon':
            transitions_info.append(Transition(type="start_codon", start_position=segment.start, end_position=segment.start + 2, frame=segment.frame))
        if segment.end_type == 'stop_codon':
            transitions_info.append(Transition(type="stop_codon", start_position=segment.end - 2, end_position=segment.end, frame=segment.frame))

    for i, transition in enumerate(transitions_info):
        attributes = [f"ID={transition.type}_{read_name}_{i}"]
        attr_string = ";".join(attributes)
        outfile_gff.write(
            f"{read_name}\tDeepCDS\t{transition.type}\t{transition.start_position}\t{transition.end_position}\t"
            f".\t+\t.\t{attr_string}\n"
        )

    for region in uncertain_regions:
        attributes = []
        attributes.append(f"Note=Uncertain region: {region.reason}")
        attributes.append(f"overlapping_frames={','.join(map(str, region.overlapping_frames))}")
        attr_string = ";".join(attributes)
        outfile_gff.write(
            f"{read_name}\tDeepCDS\tuncertain_region\t{region.start}\t{region.end}\t"
            f".\t+\t.\t{attr_string}\n"
        )


def process_predictions(predictions_rf0, predictions_rf1, predictions_rf2,
                        read_names, gff_buffers, count, min_cds_length):
    """
    Postprocess decoded predictions and write GFF output to per-sequence buffers.

    Args:
        - predictions_rf0, predictions_rf1, predictions_rf2: Lists of predicted labels for each reading frame
        - read_names: List of sequence names corresponding to the predictions
        - gff_buffers: Dictionary mapping read names to their corresponding GFF output buffers
        - count: Number of sequences in the current batch (used for progress tracking)
        - min_cds_length: Minimum length for predicted CDS sequences
    """
    for i in range(count):
        segments, uncertain_regions, transitions_info, _ = get_cds_coords(
            predictions_rf0[i], predictions_rf1[i], predictions_rf2[i])
        
        write_gff(segments, uncertain_regions, transitions_info, read_names[i], gff_buffers[read_names[i]], min_cds_length)


# ══════════════════════════════════════════════════════════════════════════════
# Inference — Short Sequences: 300 nt or shorter (direct)
# ══════════════════════════════════════════════════════════════════════════════

def run_direct_inference(model, df, mapping_dict_to_class, max_aa_len,
                         device, dtype, batch_size, num_workers_cpu, pin_memory, gff_buffers, 
                         min_cds_length):
    """
    Run direct (non-sliding-window) inference on sequences that fit within the trained window.
    
    Args: 
    - model: Loaded DeepCDS model
    - df: DataFrame containing sequences and metadata for the short sequence group
    - mapping_dict_to_class: Dict mapping model output indices to class labels
    - max_aa_len: Maximum amino acid length for padding sequences in this group
    - device: Computation device (CPU, CUDA, etc.)
    - dtype: Data type for model inputs (e.g. torch.float16)
    - batch_size: Batch size for inference
    - num_workers_cpu: Number of CPU workers for data loading
    - pin_memory: Whether to use pinned memory for DataLoader
    - gff_buffers: Dictionary of GFF buffers for each read
    """

    tokenizer = get_tokenizer()
    dataset = encode_data(df, max_aa_len, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers_cpu, pin_memory=pin_memory)

    with torch.inference_mode():
        model.eval()
        for batch in tqdm(loader, desc="Predicting on the short sequences..."):
            aa_rf0 = batch['aa_encodings_rf0']['input_ids'].to(device)
            mask_rf0 = batch['aa_encodings_rf0']['attention_mask'].to(device)
            aa_rf1 = batch['aa_encodings_rf1']['input_ids'].to(device)
            mask_rf1 = batch['aa_encodings_rf1']['attention_mask'].to(device)
            aa_rf2 = batch['aa_encodings_rf2']['input_ids'].to(device)
            mask_rf2 = batch['aa_encodings_rf2']['attention_mask'].to(device)

            nt_rf0 = batch['nt_encodings_rf0'].to(device, dtype=dtype)
            nt_rf1 = batch['nt_encodings_rf1'].to(device, dtype=dtype)
            nt_rf2 = batch['nt_encodings_rf2'].to(device, dtype=dtype)

            read_names = batch['read_name']

            # Model outputs shared label class sequence 
            outputs = model(
                nt_rf0, aa_rf0, mask_rf0,
                nt_rf1, aa_rf1, mask_rf1,
                nt_rf2, aa_rf2, mask_rf2)

            predictions_encoded = outputs["predictions"]

            # Map shared label across RFs to class label and separate into per-RF predictions
            preds_rf0, preds_rf1, preds_rf2 = [], [], []
            for preds_sample in predictions_encoded:
                preds = [mapping_dict_to_class[p] for p in preds_sample]
                preds_rf0.append([rf[0] for rf in preds])
                preds_rf1.append([rf[1] for rf in preds])
                preds_rf2.append([rf[2] for rf in preds])

            # Trim prediction sequences to actual sequence length based on EOS token in input_ids for each RF
            preds_rf0 = trim_predictions_by_eos(preds_rf0, aa_rf0)
            preds_rf1 = trim_predictions_by_eos(preds_rf1, aa_rf1)
            preds_rf2 = trim_predictions_by_eos(preds_rf2, aa_rf2)

            # Write CDS predictions to GFF buffers for each sequence in the batch
            process_predictions(preds_rf0, preds_rf1, preds_rf2,
                                read_names, gff_buffers, len(read_names), min_cds_length)

            # Cleanup to free memory after each batch
            del aa_rf0, aa_rf1, aa_rf2, mask_rf0, mask_rf1, mask_rf2
            del nt_rf0, nt_rf1, nt_rf2, outputs, predictions_encoded

    clear_memory(sync=True)


# ══════════════════════════════════════════════════════════════════════════════
# Inference — Long Sequences, longer than 300 nt (sliding window, variable length)
# ══════════════════════════════════════════════════════════════════════════════

def run_sliding_window_single(model, name, seq, mapping_dict_to_class,
                               device, dtype, batch_size, stride_aa,
                               num_workers_cpu, pin_memory, gff_buffers,
                               min_cds_length):
    """Run sliding window inference on a single long sequence."""
    tokenizer = get_tokenizer()
    seq_len = len(seq)
    window_size_aa = TRAINED_WINDOW_SIZE_AA
    window_size_nt = window_size_aa * 3
    stride_nt = stride_aa * 3

    window_starts = get_window_positions(seq_len, window_size_nt, stride_nt)
    n_windows = len(window_starts)
    full_aa_len = seq_len // 3
    num_labels = model.linear_transform.out_features

    # Create single-row DataFrame for windowing
    df = sequences_to_dataframe([name], [seq])
    windowed_df = _create_windowed_dataframe(df, window_starts, window_size_nt)
    window_dataset = encode_data(windowed_df, window_size_aa, tokenizer)

    # Run model on all windows
    all_logits = _run_model_on_windows(
        model, window_dataset, n_windows, device, dtype,
        batch_size=batch_size,
        num_workers_cpu=num_workers_cpu,
        pin_memory=pin_memory,
    )

    # Reshape: (1, n_windows, window_size_aa, num_labels)
    all_logits = all_logits.view(1, n_windows, window_size_aa, num_labels)

    # Merge shared label space logits from overlapping window positions
    merged_logits, merged_mask = _merge_window_logits(
        all_logits, window_starts, window_size_aa, full_aa_len, num_labels, device
    )

    if dtype == torch.float16:
        merged_logits = merged_logits.half()
    merged_mask = merged_mask.bool()

    # CRF decoding
    predictions_encoded = model.CRF.crf.decode(merged_logits, mask=merged_mask)

    # Decode per-RF predictions
    preds_rf0, preds_rf1, preds_rf2 = _decode_predictions(
        predictions_encoded, mapping_dict_to_class, seq_len
    )

    process_predictions(preds_rf0, preds_rf1, preds_rf2, [name], gff_buffers, 1, min_cds_length)

    del all_logits, merged_logits, merged_mask, windowed_df, window_dataset

# ══════════════════════════════════════════════════════════════════════════════
# Write sequences to fasta file
# ══════════════════════════════════════════════════════════════════════════════

def extract_cds_from_gff(fasta_path, gff_path, output_path):
    sequences = SeqIO.index(fasta_path, "fasta")

    # Collect grouped (indel-interrupted) and ungrouped CDS entries
    ungrouped = []
    groups = defaultdict(list)  # group_id -> list of GFF field rows, in order

    try:
        with open(gff_path) as gff_f:
            for line in gff_f:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 9 or fields[2] != "CDS":
                    continue

                attrs = dict(item.split("=") for item in fields[8].split(";"))

                #CDS fragments interrupted by indels will have group_id attribute in the format "group_X.Y" where X is the group number and Y is the fragment number within the group. 
                if "group_id" in attrs:
                    group_base = attrs["group_id"].rsplit(".", 1)[0] 
                    key = (fields[0], group_base)
                    groups[key].append((fields, attrs))
                
                # Complete CDS fragments without indels will not have group_id and will be processed separately
                else:
                    ungrouped.append((fields, attrs))

        with open(output_path, "w") as out_f:

            for fields, attrs in ungrouped:
                seq_name = fields[0]
                start, end, strand = int(fields[3]), int(fields[4]), fields[6]
                cds_seq = sequences[seq_name].seq[start - 1 : end]
                out_f.write(f">{seq_name}_{start}_{end}_{strand}\n{cds_seq}\n")

            for (seq_name, group_id), members in groups.items(): 
                members.sort(key=lambda x: int(x[0][3]))

                strand     = members[0][0][6]
                indel_type = members[0][1]["indel_type"]

                group_start = int(members[0][0][3])
                group_end   = int(members[-1][0][4])

                fragments = [
                    sequences[seq_name].seq[int(f[3]) - 1 : int(f[4])]
                    for f, _ in members
                ]

                #Insertion is removed
                if indel_type == "insertion":
                    merged_seq = "".join(str(f) for f in fragments)
                
                #An NNN gap is inserted to represent the deleted region, as the model cannot predict the exact sequence of the deleted region
                elif indel_type == "deletion":
                    merged_seq = "NNN".join(str(f) for f in fragments)

                header = f">{seq_name}_{group_start}_{group_end}_{strand}"
                out_f.write(f"{header}\n{merged_seq}\n")

    except ValueError as e:
        print(f"Error processing GFF file: {e}")
        sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Validate input ──────────────────────────────────────────────────────
    if not os.path.isfile(args.fasta):
        print(f"Error: FASTA file not found: {args.fasta}")
        sys.exit(1)

    # ── Output path ─────────────────────────────────────────────────────────
    if args.output is None:
        fasta_stem = os.path.splitext(os.path.basename(args.fasta))[0]
        args.output = f"{fasta_stem}_deepcds_predictions"

    # ── Model configuration ─────────────────────────────────────────────────
    # For indel+substitution model, we have 6 classes (0-5) to capture indel transitions. For the others, we have 4 classes (0-3).
    label_classes = 6 if args.error_type == "indel_substitution" else 4

    # ── Device setup ────────────────────────────────────────────────────────
    if args.compute_device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_workers_cpu = 2
        pin_memory = True
    elif args.compute_device == "mps":
        device = torch.device("mps" if torch.mps.is_available() else "cpu")
        num_workers_cpu = 0
        pin_memory = False
    else:
        device = torch.device("cpu")
        num_workers_cpu = 0
        pin_memory = False

    device_type = device.type
    print(f"Running on device: {device}")

    # ── Parse FASTA ─────────────────────────────────────────────────────────
    print(f"Reading FASTA: {args.fasta}")
    sequences = parse_fasta(args.fasta)
    print(f"  Parsed {len(sequences)} sequences")
    sequences = validate_sequences(sequences)
    if not sequences:
        print("Error: No valid sequences found.")
        sys.exit(1)
    print(f"  {len(sequences)} valid sequences")

    # ── Load model ──────────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))

    error_type_to_name = {
        "none": "deepcds",
        "substitution": "deepcds_S",
        "indel_substitution": "deepcds_SI",
    }
    model_name = error_type_to_name[args.error_type]
    ckpt_path = os.path.join(script_dir, "models", f"{model_name}.pth")
    hyperparams_path = os.path.join(script_dir, "configs", model_name, "hyperparameters.yaml")

    # Load label mapping for the specific error type (used to decode model outputs into class labels)
    label_mapping_path = os.path.join(script_dir, "configs", model_name, "label_mapping.pkl")

    if not os.path.isfile(label_mapping_path):
        print(f"Error: Label mapping not found: {label_mapping_path}")
        sys.exit(1)
    if not os.path.isfile(hyperparams_path):
        print(f"Error: Hyperparameters not found: {hyperparams_path}")
        sys.exit(1)

    """
    esm2_model_name = "facebook/esm2_t6_8M_UR50D"

    print(f"Loading DeepCDS (error_type: {args.error_type})")

    model, mapping_dict_to_class = load_model(
        ckpt_path=ckpt_path,
        label_mapping_path=label_mapping_path,
        hyperparams_path=hyperparams_path,
        device=device,
        esm2_model=esm2_model_name,
        label_classes=label_classes)

    # Half precision for GPU/MPS
    use_half = device_type in ("cuda", "mps")
    if use_half:
        model = model.half()
        dtype = torch.float16
        print(f"Using half precision (FP16) on {device_type}")
    else:
        dtype = torch.float32

    # ── Split sequences into short vs long ──────────────────────────────────
    trained_window_nt = 300

    # Preserve original FASTA order: record (name, seq, original_index, is_long)
    input_order = []  # list of (name, original_index)
    short_names, short_seqs = [], []
    long_names, long_seqs = [], []

    for idx, (name, seq) in enumerate(sequences):
        input_order.append((name, idx))
        if len(seq) <= trained_window_nt:
            short_names.append(name)
            short_seqs.append(seq)
        else:
            long_names.append(name)
            long_seqs.append(seq)

    print(f"\nSequences: {len(short_names)} short (≤{trained_window_nt} nt), "
          f"{len(long_names)} long (>{trained_window_nt} nt)")

    # ── Run inference ───────────────────────────────────────────────────────
    # Collect GFF lines per sequence into buffers, then write in original order
    gff_buffers = {}  # name -> StringIO

    with torch.inference_mode():
        model.eval()

        # Process short sequences in a single batch (padded to max length in group)
        if short_seqs:
            print(f"\nProcessing {len(short_seqs)} short sequences...")

            # Determine max sequence length in short sequence group for padding. 
            max_seq_len = max(len(s) for s in short_seqs)
            # Ensure that we pad enough to cover the longest sequence and allow for some extra padding to reach the next multiple of 3 codons for each reading frame. See "Supplementary Note X. Inference on sequence ends".
            max_aa_len = int(np.ceil(max_seq_len / 3)) + 3 
            df_short = sequences_to_dataframe(short_names, short_seqs)

            # Create per-sequence buffers
            for n in short_names:
                gff_buffers[n] = io.StringIO()

            run_direct_inference(
                model, df_short, mapping_dict_to_class, max_aa_len,
                device, dtype, args.batch_size, num_workers_cpu, pin_memory, gff_buffers, args.min_cds_length
            )
            del df_short
            clear_memory()

        # Process long sequences individually with sliding window: See "Supplementary Note X. Inference on longer sequences"
        if long_seqs:
            print(f"\nProcessing {len(long_seqs)} long sequences with sliding window...")
            for name, seq in tqdm(zip(long_names, long_seqs), total=len(long_seqs), desc="Long sequences"):
                gff_buffers[name] = io.StringIO()
                run_sliding_window_single(
                    model, name, seq, mapping_dict_to_class,
                    device, dtype, args.batch_size, args.stride_aa,
                    num_workers_cpu, pin_memory, gff_buffers, args.min_cds_length
                )
                clear_memory()

    # Write GFF output in original FASTA order
    with open(f"{args.output}.gff", "w") as outfile_gff:
        outfile_gff.write("##gff-version 3\n")
        for name, _ in input_order:
            if name in gff_buffers:
                outfile_gff.write(gff_buffers[name].getvalue())

    """

    extract_cds_from_gff(args.fasta, f"{args.output}.gff", f"{args.output}.fna")

    clear_memory(sync=True)

    print(f"\nDeepCDS finished succesfully!")
    print(f"\tPredicted CDS coordinates in GFF format are written to: {args.output}.gff")
    print(f"\tPredicted CDS sequences in FASTA format are written to: {args.output}.fna")


if __name__ == "__main__":
    main()
