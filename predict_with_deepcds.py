"""
DeepCDS Prediction Script — User-facing

Predict coding sequences (CDS) in nucleotide FASTA sequences using trained
DeepCDS models. Supports variable-length input sequences and three model
variants trained on different error profiles.

Usage examples:
    # No sequencing errors (clean sequences)
    python predict_with_deepcds.py --input_fasta input.fasta --error_model none --output predictions.gff

    # Sequences with substitution errors (e.g. Illumina)
    python predict_with_deepcds.py --input_fasta input.fasta --error_model S --output predictions.gff

    # Sequences with indel + substitution errors
    python predict_with_deepcds.py --input_fasta input.fasta --error_model SI --output predictions.gff
"""

import argparse
import gc
import logging
import os
import gzip
import io
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional
import csv

# Suppress noisy third-party library warnings (must run before transformers import)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message="enable_nested_tensor", category=UserWarning)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src import TRAINED_WINDOW_SIZE_AA, encode_data, load_model, extract_cds_from_gff, reverse_complement
from src.sliding_window import (
    _create_windowed_dataframe,
    _decode_predictions,
    _merge_window_logits,
    _run_model_on_windows,
    get_window_positions,
)

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
  python predict_with_deepcds.py --input_fasta reads.fasta --error_model none
  python predict_with_deepcds.py --input_fasta reads.fasta --error_model S --output my_predictions
  python predict_with_deepcds.py --input_fasta reads.fasta --error_model SI --batch_size 128
        """,
    )
    parser.add_argument(
        "-in", "--input_fasta",
        type=str,
        required=True,
        help="Path to input FASTA file with nucleotide sequences (can also be passed in gzipped format with .gz extension)",
    )
    parser.add_argument(
        "--error_model",
        type=str,
        required=True,
        choices=["none", "S", "SI"],
        help=(
            "Error profile the DeepCDS model version was trained on: "
            "- 'none' for error-free sequences (runs the DeepCDS (Full) model)"
            "- 'S' for sequences with substitution errors (runs the DeepCDS S (Full) model)"
            "- 'SI' for sequences with substitution, insertion, and deletion errors (runs theDeepCDS S+I (Full) model)"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path and name without file format extension (default: <fasta_stem>_deepcds_predictions)",
    )
    parser.add_argument('--compute_device',
                        type=str,
                        default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help='Hardware accelerator to use. "auto" (default) selects the best available device (cuda → mps → cpu). Other options: "cuda" (NVIDIA GPU), "mps" (Apple Silicon), "cpu".')

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for inference (how many sequences are processed together in one iteration). If you have limited memory, try a smaller batch size (default: 128)",
    )

    parser.add_argument(
        "--min_cds_length",
        type=int,
        default=60,
        help="The minimum length that predicted CDS sequences can have. We recommend not going below 30 nt as this may affect prediction accuracy to a large extent (default: 60)",
    )

    parser.add_argument(
        "--stride_aa",
        type=int,
        default=50,
        help="Sliding window stride in codons for long sequences (how many codons the prediction window advances between each inference step). Smaller stride gives larger overlap between consecutive windows and may improve accuracy, but increases computation time (default: 50)",
    )
    
    parser.add_argument(
        "--gzip_output",
        action="store_true",
        help="Compress output files (.gff.gz, .fna.gz, .faa.gz) with gzip",
    )

    parser.add_argument(
        "--suppress_output_files",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=[],
        help="Comma-separated list of output formats to suppress. Choices: gff, fna, faa (e.g. --suppress_output_files fna,faa)",
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

    _open = gzip.open if fasta_path.endswith(".gz") else open
    with _open(fasta_path, "rt") as f:
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
            print(f"  Warning: Skipping '{name}' - sequence too short ({len(seq)} nt, minimum 30 nt)")
            continue
        invalid_chars = set(seq) - valid_nucs
        if invalid_chars:
            print(f"  Warning: '{name}' contains non-standard characters: {invalid_chars} - treating as N")

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

def write_gff(segments, uncertain_regions, transitions_info, read_name, outfile_gff,
              min_cds_length, strand="+", seq_len=None):
    """Write CDS predictions to GFF file.

    For complement-strand predictions, segment coordinates are RC-space coordinates.
    seq_len is required to convert them to forward-strand GFF coordinates:
        gff_start = seq_len - segment.end   + 1
        gff_end   = seq_len - segment.start + 1
    """
    counter_cds_frags_interrupted = {}
    cds_n = 0

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

        if segment.group_id:
            group_num = segment.group_id.split("_")[1]
            cds_id = f"{read_name}_group{strand}_{group_num}"
        else:
            cds_n += 1
            cds_id = f"{read_name}_CDS{strand}_{cds_n}"
        attributes.insert(0, f"ID={cds_id}")

        # Convert RC coordinates to forward-strand GFF coordinates for complement strand
        if strand == "-":
            gff_start = seq_len - segment.end   + 1
            gff_end   = seq_len - segment.start + 1
        else:
            gff_start = segment.start
            gff_end   = segment.end

        attr_string = ";".join(attributes)
        outfile_gff.write(
            f"{read_name}\tDeepCDS\tCDS\t{gff_start}\t{gff_end}\t"
            f".\t{strand}\t{segment.frame}\t{attr_string}\n"
        )

        # start_codon sits at the 5' end of the CDS (gff_end side on complement strand)
        if segment.start_type == 'start_codon':
            if strand == "+":
                transitions_info.append(Transition(type="start_codon",
                    start_position=gff_start, end_position=gff_start + 2, frame=segment.frame))
            else:
                transitions_info.append(Transition(type="start_codon",
                    start_position=gff_end - 2, end_position=gff_end, frame=segment.frame))

        # stop_codon sits at the 3' end of the CDS (gff_start side on complement strand)
        if segment.end_type == 'stop_codon':
            if strand == "+":
                transitions_info.append(Transition(type="stop_codon",
                    start_position=gff_end - 2, end_position=gff_end, frame=segment.frame))
            else:
                transitions_info.append(Transition(type="stop_codon",
                    start_position=gff_start, end_position=gff_start + 2, frame=segment.frame))

    for i, transition in enumerate(transitions_info):
        attributes = [f"ID={transition.type}_{read_name}_{i}"]
        attr_string = ";".join(attributes)
        outfile_gff.write(
            f"{read_name}\tDeepCDS\t{transition.type}\t{transition.start_position}\t{transition.end_position}\t"
            f".\t{strand}\t.\t{attr_string}\n"
        )

    for region in uncertain_regions:
        if strand == "-":
            r_start = seq_len - region.end   + 1
            r_end   = seq_len - region.start + 1
        else:
            r_start = region.start
            r_end   = region.end
        attributes = []
        attributes.append(f"Note=Uncertain region: {region.reason}")
        attributes.append(f"overlapping_frames={','.join(map(str, region.overlapping_frames))}")
        attr_string = ";".join(attributes)
        outfile_gff.write(
            f"{read_name}\tDeepCDS\tuncertain_region\t{r_start}\t{r_end}\t"
            f".\t{strand}\t.\t{attr_string}\n"
        )


def process_predictions(predictions_rf0, predictions_rf1, predictions_rf2,
                        read_names, gff_buffers, count, min_cds_length,
                        strand="+", seq_lengths=None):
    """
    Postprocess decoded predictions and write GFF output to per-sequence buffers.

    Args:
        - predictions_rf0, predictions_rf1, predictions_rf2: Lists of predicted labels for each reading frame
        - read_names: List of sequence names corresponding to the predictions
        - gff_buffers: Dictionary mapping read names to their corresponding GFF output buffers
        - count: Number of sequences in the current batch (used for progress tracking)
        - min_cds_length: Minimum length for predicted CDS sequences
        - strand: "+" or "-". For complement strand, seq_lengths must be provided for coordinate conversion.
        - seq_lengths: Dict mapping read_name to original sequence length (required for strand="-").
    """
    for i in range(count):
        seq_len = seq_lengths[read_names[i]] if seq_lengths is not None else None
        segments, uncertain_regions, transitions_info, _ = get_cds_coords(
            predictions_rf0[i], predictions_rf1[i], predictions_rf2[i])

        write_gff(segments, uncertain_regions, transitions_info, read_names[i],
                  gff_buffers[read_names[i]], min_cds_length,
                  strand=strand, seq_len=seq_len)


# ══════════════════════════════════════════════════════════════════════════════
# Inference — Short Sequences: 300 nt or shorter (direct)
# ══════════════════════════════════════════════════════════════════════════════

def run_direct_inference(model, df, mapping_dict_to_class, max_aa_len,
                         device, dtype, batch_size, num_workers_cpu, pin_memory, gff_buffers,
                         min_cds_length, strand="+", seq_lengths=None):
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
        for batch in tqdm(loader, desc="Predicting on the short sequences...", file=sys.stdout):
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
                                read_names, gff_buffers, len(read_names), min_cds_length,
                                strand=strand, seq_lengths=seq_lengths)

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
                               min_cds_length, strand="+"):
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

    process_predictions(preds_rf0, preds_rf1, preds_rf2, [name], gff_buffers, 1, min_cds_length,
                        strand=strand, seq_lengths={name: seq_len})

    del all_logits, merged_logits, merged_mask, windowed_df, window_dataset

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Validate arguments ──────────────────────────────────────────────────
    _valid_formats = {"gff", "fna", "faa"}
    _invalid = set(args.suppress_output_files) - _valid_formats
    if _invalid:
        print(f"Error: --suppress_output_files: invalid format(s): {', '.join(sorted(_invalid))}. Choose from: gff, fna, faa")
        sys.exit(1)

    suppressed = set(args.suppress_output_files)

    if not os.path.isfile(args.input_fasta):
        print(f"Error: FASTA file not found: {args.input_fasta}")
        sys.exit(1)

    # ── Output path ─────────────────────────────────────────────────────────
    if args.output is None:
        fasta_stem = os.path.basename(args.input_fasta)
        while True:
            base, ext = os.path.splitext(fasta_stem)
            if ext in (".gz", ".fasta", ".fa", ".fna", ".fq", ".fastq"):
                fasta_stem = base
            else:
                break
        args.output = f"{fasta_stem}_deepcds_predictions"

    # ── Model configuration ─────────────────────────────────────────────────
    # For indel+substitution model, we have 6 classes (0-5) to capture indel transitions. For the others, we have 4 classes (0-3).
    label_classes = 6 if args.error_model == "SI" else 4

    # ── Device setup ────────────────────────────────────────────────────────
    def _resolve_device(requested: str) -> torch.device:
        if requested in ("auto", "cuda"):
            if torch.cuda.is_available():
                return torch.device("cuda")
            if requested == "auto" and torch.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if requested == "mps":
            return torch.device("mps" if torch.mps.is_available() else "cpu")
        return torch.device("cpu")

    device = _resolve_device(args.compute_device)
    num_workers_cpu = 2 if device.type == "cuda" else 0
    pin_memory      = device.type == "cuda"

    device_type = device.type
    print(f"Running on device: {device}")

    # ── Parse FASTA ─────────────────────────────────────────────────────────
    print(f"Reading FASTA: {args.input_fasta}")
    sequences = parse_fasta(args.input_fasta)
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
        "S": "deepcds_S",
        "SI": "deepcds_SI",
    }
    model_name = error_type_to_name[args.error_model]
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

    esm2_model_name = "facebook/esm2_t6_8M_UR50D"

    print(f"Loading DeepCDS (error_type: {args.error_model})")

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

    print(f"\nSequences: {len(short_names)} short (<={trained_window_nt} nt), "
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
            for name, seq in tqdm(zip(long_names, long_seqs), total=len(long_seqs), desc="Long sequences", file=sys.stdout):
                gff_buffers[name] = io.StringIO()
                run_sliding_window_single(
                    model, name, seq, mapping_dict_to_class,
                    device, dtype, args.batch_size, args.stride_aa,
                    num_workers_cpu, pin_memory, gff_buffers, args.min_cds_length
                )
                clear_memory()

    # ── Run inference on reverse complement sequences (complement strand) ─────────
    # seq_lengths maps each read name to its original length, needed to convert
    # RC coordinates back to forward-strand GFF coordinates.
    seq_lengths = {name: len(seq) for name, seq in sequences}

    rc_short_names, rc_short_seqs = [], []
    rc_long_names,  rc_long_seqs  = [], []
    for name, seq in sequences:
        rc_seq = reverse_complement(seq)
        if len(rc_seq) <= trained_window_nt:
            rc_short_names.append(name)
            rc_short_seqs.append(rc_seq)
        else:
            rc_long_names.append(name)
            rc_long_seqs.append(rc_seq)

    with torch.inference_mode():
        model.eval()

        if rc_short_seqs:
            print(f"\nProcessing {len(rc_short_seqs)} short sequences (complement strand)...")
            max_seq_len_rc = max(len(s) for s in rc_short_seqs)
            max_aa_len_rc  = int(np.ceil(max_seq_len_rc / 3)) + 3
            df_rc_short = sequences_to_dataframe(rc_short_names, rc_short_seqs)

            run_direct_inference(
                model, df_rc_short, mapping_dict_to_class, max_aa_len_rc,
                device, dtype, args.batch_size, num_workers_cpu, pin_memory,
                gff_buffers, args.min_cds_length,
                strand="-", seq_lengths=seq_lengths
            )
            del df_rc_short
            clear_memory()

        if rc_long_seqs:
            print(f"\nProcessing {len(rc_long_seqs)} long sequences with sliding window (complement strand)...")
            for name, rc_seq in tqdm(zip(rc_long_names, rc_long_seqs),
                                     total=len(rc_long_seqs), desc="Long sequences (complement strand)", file=sys.stdout):
                run_sliding_window_single(
                    model, name, rc_seq, mapping_dict_to_class,
                    device, dtype, args.batch_size, args.stride_aa,
                    num_workers_cpu, pin_memory, gff_buffers, args.min_cds_length,
                    strand="-"
                )
                clear_memory()

    # Write GFF output in original FASTA order
    ext = ".gz" if args.gzip_output else ""
    want_gff = "gff" not in suppressed
    want_fna = "fna" not in suppressed
    want_faa = "faa" not in suppressed

    gff_path = f"{args.output}.gff{ext}" if want_gff else None
    fna_path = f"{args.output}.fna{ext}" if want_fna else None
    faa_path = f"{args.output}.faa{ext}" if want_faa else None

    open_fn = gzip.open if args.gzip_output else open

    # If GFF is suppressed but fna/faa are needed, write GFF to a temp file
    _tmp_gff = None
    if not want_gff and (want_fna or want_faa):
        import tempfile
        _tmp = tempfile.NamedTemporaryFile(mode="wt", suffix=".gff", delete=False)
        _tmp.write("##gff-version 3\n")
        for name, _ in input_order:
            if name in gff_buffers:
                _tmp.write(gff_buffers[name].getvalue())
        _tmp.close()
        _tmp_gff = _tmp.name
    elif want_gff:
        with open_fn(gff_path, "wt") as outfile_gff:
            outfile_gff.write("##gff-version 3\n")
            for name, _ in input_order:
                if name in gff_buffers:
                    outfile_gff.write(gff_buffers[name].getvalue())

    if want_fna or want_faa:
        extract_cds_from_gff(
            args.input_fasta,
            _tmp_gff if _tmp_gff else gff_path,
            fna_path,
            faa_path,
        )
        if _tmp_gff:
            os.remove(_tmp_gff)

    clear_memory(sync=True)

    print(f"\nDeepCDS finished succesfully!")
    if want_gff:
        print(f"\tPredicted CDS coordinates in GFF format are written to: {gff_path}")
    if want_fna:
        print(f"\tPredicted CDS sequences in FASTA format are written to: {fna_path}")
    if want_faa:
        print(f"\tPredicted CDS sequences (translated) are written to: {faa_path}")


if __name__ == "__main__":
    main()
