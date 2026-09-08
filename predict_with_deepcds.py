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

__version__ = "1.0.0"

# Handle --version before the heavy ML imports below, so it doesn't require
# torch/transformers/pandas to be installed just to print a version string.
if "--version" in sys.argv:
    print(f"DeepCDS v{__version__}")
    sys.exit(0)

# Suppress noisy third-party library warnings (must run before transformers import)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message="enable_nested_tensor", category=UserWarning)

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Imported from the submodules rather than through the package's re-export list, so
# this script does not depend on which names a given checkout's src/__init__.py happens
# to export - only on the modules themselves.
from src.deepcds_model import load_model
from src.fast_inference import (
    build_label_lut,
    codon_one_hot_from_codes,
    encode_reads_fast,
    viterbi_decode_fast,
)
from src.postprocessing import extract_cds_from_gff, reverse_complement
from src.sliding_window import TRAINED_WINDOW_SIZE_AA, get_window_positions

logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════════════════════
# Argument Parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Predict coding sequences (CDS) in nucleotide FASTA sequences using DeepCDS (v{__version__}).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_with_deepcds.py --input_fasta reads.fasta --error_model none
  python predict_with_deepcds.py --input_fasta reads.fasta --error_model S --output my_predictions
  python predict_with_deepcds.py --input_fasta reads.fasta --error_model SI --batch_size 128
        """,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
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
                        help='Hardware accelerator to use. "auto" (default) selects the best available device (cuda -> mps -> cpu). Other options: "cuda" (NVIDIA GPU), "mps" (Apple Silicon), "cpu".')

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
        "--chunk_size",
        type=int,
        default=100_000,
        help="Number of sequences held in memory at a time. The file is streamed in "
             "chunks of this size, so peak memory stays bounded regardless of input size "
             "(roughly 1.3 kB per sequence held). Inputs smaller than one chunk are "
             "processed exactly as before. Does not affect predictions or runtime, only "
             "memory (default: 100000)",
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

def iter_fasta_records(fasta_path):
    """
    Yield (name, sequence) tuples from a FASTA file one record at a time.

    Handles multi-line sequences and strips whitespace. Sequence names are
    taken from the first word of the header line (after '>').
    """
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
                    yield current_name, "".join(current_seq_parts).upper()
                current_name = line[1:].split()[0]
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

    # Don't forget the last sequence
    if current_name is not None:
        yield current_name, "".join(current_seq_parts).upper()


def count_fasta_sequences(fasta_path):
    """Count '>' records without parsing, for the progress bar's total.

    One extra pass over the file, in 1 MB blocks, counting header lines. That is cheap
    next to inference and is what lets the bar show a meaningful ETA from the first
    batch rather than only a rate.
    """
    _open = gzip.open if fasta_path.endswith(".gz") else open
    count = 0
    at_line_start = True
    with _open(fasta_path, "rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            if at_line_start and block.startswith(b">"):
                count += 1
            count += block.count(b"\n>")
            at_line_start = block.endswith(b"\n")
    return count


def iter_fasta_chunks(fasta_path, chunk_size):
    """Yield lists of at most `chunk_size` (name, sequence) tuples.

    Streaming in chunks is what keeps peak memory independent of input size: the
    script holds each sequence, its reverse complement and a GFF buffer until that
    sequence's output is written, which is roughly 1.3 kB per sequence. Bounding
    how many are alive at once bounds the total.
    """
    chunk = []
    for record in iter_fasta_records(fasta_path):
        chunk.append(record)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def parse_fasta(fasta_path):
    """
    Parse a FASTA file and return a list of (name, sequence) tuples.

    Reads the whole file into memory; prefer iter_fasta_chunks() for large inputs.
    """
    return list(iter_fasta_records(fasta_path))


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


def _frames_to_device(encoded, begin, stop, device, dtype):
    """Move one slice of an encode_reads_fast() result onto the device.

    Token ids and masks cross the bus as int16/int8 and are widened on the device;
    nucleotides cross as one byte per base and are expanded into the (L, 12) codon
    one-hot there, which is ~48x less traffic than shipping the float one-hot that
    the old DataLoader collated on the host.
    """
    aa_frames, mask_frames, nt_frames = [], [], []
    for rf in range(3):
        aa_frames.append(
            torch.from_numpy(encoded["input_ids"][rf, begin:stop]).to(device).long()
        )
        mask_frames.append(
            torch.from_numpy(encoded["attention_mask"][rf, begin:stop]).to(device).long()
        )
        codes = torch.from_numpy(encoded["nt_codes"][rf, begin:stop]).to(device)
        nt_frames.append(codon_one_hot_from_codes(codes, dtype=dtype))
    return nt_frames, aa_frames, mask_frames


def _decode_to_rf_labels(model, label_lut, logits, combined_mask, trim_lengths):
    """CRF-decode a batch and split it into per-reading-frame label lists.

    Replaces three separate slow steps: torchcrf's Viterbi backtrace (which reads one
    value off the GPU per token per sequence, each read forcing a full device
    synchronisation), the per-token dict lookup into mapping_dict_to_class, and
    trim_predictions_by_eos (which located the EOS token with a GPU op per sequence).
    All three now cost one host transfer and two vectorised NumPy operations.

    Args:
        trim_lengths: (3, batch) array of per-frame lengths to cut each prediction to,
            i.e. what trim_predictions_by_eos derived from the EOS position.

    Returns:
        (preds_rf0, preds_rf1, preds_rf2), each a list of per-sequence int lists.
    """
    tags, lengths = viterbi_decode_fast(model.CRF.crf, logits.float(), combined_mask.bool())

    tags_np = tags.cpu().numpy()        # the single host sync for the whole batch
    lengths_np = lengths.cpu().numpy()

    # (3, B, L) per-frame labels in one fancy-index plus one .tolist()
    rf_labels = np.ascontiguousarray(label_lut[tags_np].transpose(2, 0, 1))
    preds_rf0, preds_rf1, preds_rf2 = rf_labels.tolist()

    # The CRF only decoded `lengths` positions; each frame is then cut to its own end.
    trims = np.minimum(trim_lengths, lengths_np[None, :])
    for rf, preds in enumerate((preds_rf0, preds_rf1, preds_rf2)):
        trim_rf = trims[rf]
        for i in range(len(preds)):
            preds[i] = preds[i][:trim_rf[i]]
    return preds_rf0, preds_rf1, preds_rf2


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

def run_direct_inference(model, names, seqs, label_lut, device, dtype, batch_size,
                         gff_buffers, min_cds_length, strand="+", seq_lengths=None,
                         pbar=None):
    """
    Run direct (non-sliding-window) inference on sequences that fit within the trained window.

    Two changes from the original beyond the shared fast decode path:

    * Sequences are batched in length order, and each batch is padded only to its own
      longest member instead of to the longest sequence in the whole file. A file mixing
      60 nt and 300 nt sequences previously padded every one of them to 300 nt and paid
      up to 5x the necessary compute. Padding beyond a sequence's own length is masked
      out of ESM-2 attention, the transformer encoder and the CRF, so this changes no
      prediction - the one thing that must hold is that every sequence keeps at least one
      codon of padding, which the +3 buffer below guarantees, because that is what keeps
      each frame's EOS token inside the trimmed attention window.
    * encode_reads_fast replaces encode_data + DataLoader, dropping the per-sequence ESM
      tokenizer walk (a pure-Python character Trie) and the per-sequence tensor
      allocations.

    Output is unaffected by the batching order: GFF lines go into per-sequence buffers
    and are written out in the original FASTA order by the caller.
    """
    if not seqs:
        return

    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))

    with torch.inference_mode():
        model.eval()
        for begin in range(0, len(order), batch_size):
            idx = order[begin:begin + batch_size]
            batch_seqs = [seqs[i] for i in idx]
            batch_names = [names[i] for i in idx]

            batch_max_len = max(len(x) for x in batch_seqs)
            # Same formula the original applied globally, now per batch. See
            # "Supplementary Note X. Inference on sequence ends".
            max_aa_len = int(np.ceil(batch_max_len / 3)) + 3

            encoded = encode_reads_fast(batch_seqs, max_aa_len, read_len=batch_max_len)
            nt_frames, aa_frames, mask_frames = _frames_to_device(
                encoded, 0, len(batch_seqs), device, dtype
            )

            # All three reading frames in one batched pass through the shared stack.
            logits, combined_mask = model.predict_logits(nt_frames, aa_frames, mask_frames)

            preds_rf0, preds_rf1, preds_rf2 = _decode_to_rf_labels(
                model, label_lut, logits, combined_mask, encoded["trim_lengths"]
            )

            process_predictions(preds_rf0, preds_rf1, preds_rf2,
                                batch_names, gff_buffers, len(batch_names), min_cds_length,
                                strand=strand, seq_lengths=seq_lengths)

            # Each sequence needs both strands, so one strand is half of its work.
            if pbar is not None:
                pbar.update(len(batch_names) / 2)

    clear_memory(sync=True)


# ══════════════════════════════════════════════════════════════════════════════
# Inference — Long Sequences, longer than 300 nt (sliding window, variable length)
# ══════════════════════════════════════════════════════════════════════════════

def _merge_windows_varlen(window_logits, seq_offsets, seq_window_starts, full_aa_lens,
                          window_size_aa, num_labels, device):
    """Average overlapping window logits back into per-sequence tracks.

    Generalises sliding_window._merge_window_logits to a block of sequences that need
    not share a length: each sequence writes into its own row of a padded tensor, and
    the mask marks how far that row is real. Identical arithmetic - accumulate the
    windows that cover each codon, then divide by the cover count.

    Returns:
        (merged_logits, merged_mask): (n_seq, max_aa_len, K) float32 and (n_seq, max_aa_len) bool.
    """
    n_seq = len(seq_offsets)
    max_len = max(full_aa_lens)

    merged = torch.zeros(n_seq, max_len, num_labels, dtype=torch.float32, device=device)
    counts = torch.zeros(n_seq, max_len, 1, dtype=torch.float32, device=device)

    for si in range(n_seq):
        offset = seq_offsets[si]
        full_aa_len = full_aa_lens[si]
        for wi, start_nt in enumerate(seq_window_starts[si]):
            start_aa = start_nt // 3
            actual_len = min(window_size_aa, full_aa_len - start_aa)
            merged[si, start_aa:start_aa + actual_len, :] += window_logits[offset + wi, :actual_len, :]
            counts[si, start_aa:start_aa + actual_len, :] += 1

    merged = merged / counts.clamp(min=1)
    return merged, (counts.squeeze(-1) > 0)


def run_sliding_window(model, names, seqs, label_lut, device, dtype, batch_size, stride_aa,
                       gff_buffers, min_cds_length, strand="+", seq_lengths=None,
                       pbar=None):
    """Run sliding-window inference over all long sequences, batched across sequences.

    The original processed one long sequence at a time, so each model call saw only that
    sequence's handful of windows and each CRF decode covered a single sequence. Every
    window is exactly the trained window size regardless of how long its parent sequence
    is, so windows from different sequences - of different lengths - batch together
    freely. Sequences are accumulated into blocks, all their windows go through the model
    in full batches, the windows are then averaged back into per-sequence tracks, and the
    CRF decodes the whole block at once over a ragged mask.
    """
    if not seqs:
        return

    window_size_aa = TRAINED_WINDOW_SIZE_AA
    window_size_nt = window_size_aa * 3
    stride_nt = stride_aa * 3
    num_labels = model.linear_transform.out_features

    # Bound how much is in flight at once; always allow at least one sequence, however
    # many windows it needs.
    window_block = max(4 * batch_size, 512)

    plans = []  # (name, seq, window_starts)
    for name, seq in zip(names, seqs):
        plans.append((name, seq, get_window_positions(len(seq), window_size_nt, stride_nt)))

    blocks, current, current_windows = [], [], 0
    for plan in plans:
        n_win = len(plan[2])
        if current and current_windows + n_win > window_block:
            blocks.append(current)
            current, current_windows = [], 0
        current.append(plan)
        current_windows += n_win
    if current:
        blocks.append(current)

    with torch.inference_mode():
        model.eval()
        for block in blocks:
            block_names = [p[0] for p in block]
            block_starts = [p[2] for p in block]
            full_aa_lens = [len(p[1]) // 3 for p in block]
            seq_lens = [len(p[1]) for p in block]

            windows, seq_offsets = [], []
            for _, seq, starts in block:
                seq_offsets.append(len(windows))
                windows.extend(seq[st:st + window_size_nt] for st in starts)

            encoded = encode_reads_fast(windows, window_size_aa, read_len=window_size_nt)

            # Model forward over every window in the block, in full batches.
            logits_parts = []
            for begin in range(0, len(windows), batch_size):
                stop = min(begin + batch_size, len(windows))
                nt_frames, aa_frames, mask_frames = _frames_to_device(
                    encoded, begin, stop, device, dtype
                )
                part, _ = model.predict_logits(nt_frames, aa_frames, mask_frames)
                logits_parts.append(part.float())
            window_logits = torch.cat(logits_parts, dim=0)  # (total_windows, window_size_aa, K)

            merged_logits, merged_mask = _merge_windows_varlen(
                window_logits, seq_offsets, block_starts, full_aa_lens,
                window_size_aa, num_labels, device,
            )

            # Per-frame prediction lengths, as _decode_predictions derived them.
            trim_lengths = np.array(
                [[sl // 3 for sl in seq_lens],
                 [(sl - 1) // 3 for sl in seq_lens],
                 [(sl - 2) // 3 for sl in seq_lens]], dtype=np.int64,
            )

            preds_rf0, preds_rf1, preds_rf2 = _decode_to_rf_labels(
                model, label_lut, merged_logits, merged_mask, trim_lengths
            )

            process_predictions(preds_rf0, preds_rf1, preds_rf2,
                                block_names, gff_buffers, len(block_names), min_cds_length,
                                strand=strand, seq_lengths=seq_lengths)

            if pbar is not None:
                pbar.update(len(block_names) / 2)

            del window_logits, merged_logits, merged_mask, logits_parts

    clear_memory(sync=True)


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

    print(f"Running on device: {device}")

    # ── Output paths ────────────────────────────────────────────────────────
    ext = ".gz" if args.gzip_output else ""
    want_gff = "gff" not in suppressed
    want_fna = "fna" not in suppressed
    want_faa = "faa" not in suppressed

    gff_path = f"{args.output}.gff{ext}" if want_gff else None
    fna_path = f"{args.output}.fna{ext}" if want_fna else None
    faa_path = f"{args.output}.faa{ext}" if want_faa else None
    open_fn = gzip.open if args.gzip_output else open

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

    # Always run inference in FP32. Reduced precision was found to measurably degrade
    # prediction quality - severely so on sequences outside the training distribution,
    # whose CRF score margins are narrow enough for rounding to flip them - and the
    # model checkpoints are themselves FP32-native.
    dtype = torch.float32

    # {encoded label -> (rf0, rf1, rf2)} as an array, so decoding a batch of predictions
    # is one fancy-index rather than a Python dict lookup per token.
    label_lut = build_label_lut(mapping_dict_to_class)

    trained_window_nt = 300

    # ── Stream the FASTA and predict chunk by chunk ─────────────────────────
    # Each chunk is fully processed (both strands) and its GFF written before the next
    # is read, so peak memory is set by --chunk_size rather than by the input size.
    # Chunks are read, and written, in input order, so the output is byte-for-byte what
    # processing the whole file at once produces.
    print(f"Reading FASTA: {args.input_fasta}")
    print("  Counting sequences...", end="", flush=True)
    n_total = count_fasta_sequences(args.input_fasta)
    print(f" {n_total}")

    n_parsed = n_valid = n_short_total = n_long_total = 0
    gff_out = None
    tmp_gff_path = None

    # One bar for the whole run, counting sequences. Each sequence is predicted on both
    # strands, so a strand pass advances it by half - which keeps the unit "sequences"
    # while still moving smoothly rather than jumping once per chunk.
    pbar = tqdm(total=n_total, unit="seq", desc="Predicting", file=sys.stdout,
                smoothing=0.05, bar_format="{l_bar}{bar}| {n:.0f}/{total_fmt} "
                                           "[{elapsed}<{remaining}, {rate_fmt}]")

    try:
        for chunk in iter_fasta_chunks(args.input_fasta, args.chunk_size):
            n_parsed += len(chunk)
            sequences = validate_sequences(chunk)
            # Sequences dropped by validation are never predicted on, so credit them
            # now; otherwise the bar could not reach its total.
            pbar.update(len(chunk) - len(sequences))
            if not sequences:
                continue
            n_valid += len(sequences)

            # Open output lazily, so an input with no valid sequences leaves no files
            # behind, exactly as the non-streaming version did.
            if gff_out is None:
                if want_gff:
                    gff_out = open_fn(gff_path, "wt")
                else:
                    import tempfile
                    _tmp = tempfile.NamedTemporaryFile(mode="wt", suffix=".gff", delete=False)
                    tmp_gff_path = _tmp.name
                    gff_out = _tmp
                gff_out.write("##gff-version 3\n")

            short_names, short_seqs = [], []
            long_names, long_seqs = [], []
            for name, seq in sequences:
                if len(seq) <= trained_window_nt:
                    short_names.append(name)
                    short_seqs.append(seq)
                else:
                    long_names.append(name)
                    long_seqs.append(seq)
            n_short_total += len(short_names)
            n_long_total += len(long_names)

            gff_buffers = {name: io.StringIO() for name, _ in sequences}
            seq_lengths = {name: len(seq) for name, seq in sequences}

            # ── Forward strand ──────────────────────────────────────────────
            if short_seqs:
                run_direct_inference(
                    model, short_names, short_seqs, label_lut,
                    device, dtype, args.batch_size, gff_buffers, args.min_cds_length,
                    pbar=pbar,
                )
            if long_seqs:
                # See "Supplementary Note X. Inference on longer sequences"
                run_sliding_window(
                    model, long_names, long_seqs, label_lut,
                    device, dtype, args.batch_size, args.stride_aa,
                    gff_buffers, args.min_cds_length, pbar=pbar,
                )

            # ── Complement strand ───────────────────────────────────────────
            # Written after the forward strand for every sequence, so each sequence's
            # GFF block keeps the same '+' then '-' ordering as before.
            rc_short_names, rc_short_seqs = [], []
            rc_long_names, rc_long_seqs = [], []
            for name, seq in sequences:
                rc_seq = reverse_complement(seq)
                if len(rc_seq) <= trained_window_nt:
                    rc_short_names.append(name)
                    rc_short_seqs.append(rc_seq)
                else:
                    rc_long_names.append(name)
                    rc_long_seqs.append(rc_seq)

            if rc_short_seqs:
                run_direct_inference(
                    model, rc_short_names, rc_short_seqs, label_lut,
                    device, dtype, args.batch_size, gff_buffers, args.min_cds_length,
                    strand="-", seq_lengths=seq_lengths, pbar=pbar,
                )
            if rc_long_seqs:
                run_sliding_window(
                    model, rc_long_names, rc_long_seqs, label_lut,
                    device, dtype, args.batch_size, args.stride_aa,
                    gff_buffers, args.min_cds_length,
                    strand="-", seq_lengths=seq_lengths, pbar=pbar,
                )

            # Flush this chunk in input order, then let it go.
            for name, _ in sequences:
                buf = gff_buffers.get(name)
                if buf is not None:
                    gff_out.write(buf.getvalue())

            del gff_buffers, seq_lengths, sequences, chunk
            del short_names, short_seqs, long_names, long_seqs
            del rc_short_names, rc_short_seqs, rc_long_names, rc_long_seqs
            clear_memory()
    finally:
        pbar.close()
        if gff_out is not None:
            gff_out.close()

    print(f"\n  Parsed {n_parsed} sequences")
    print(f"  {n_valid} valid sequences ({n_short_total} short, {n_long_total} long)")

    if n_valid == 0:
        print("Error: No valid sequences found.")
        sys.exit(1)

    if want_fna or want_faa:
        extract_cds_from_gff(
            args.input_fasta,
            tmp_gff_path if tmp_gff_path else gff_path,
            fna_path,
            faa_path,
        )
    if tmp_gff_path:
        os.remove(tmp_gff_path)

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

