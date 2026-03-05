"""
DeepCDS Prediction Script

This script runs CDS predictions using trained DeepCDS models.
"""

import argparse
import gc
import gzip
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src_dev.modeling import encode_data, load_model, sliding_window_inference, TRAINED_WINDOW_SIZE_AA

logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

# Clear the GPU memory cache
torch.cuda.empty_cache()
pd.options.mode.chained_assignment = None  # Suppress the warning globally

# Configure CUDA memory allocations (helps manage fragmentation in the GPU memory)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

################################################################################################################################
################################################Argument Parser#################################################################
################################################################################################################################

parser = argparse.ArgumentParser(description="Run DeepCDS predictions")
parser.add_argument("--gpu", type=int, default=0, help="GPU number to use (default: 0)")
parser.add_argument("--healthtech_cluster", action="store_true", help="Whether running on HealthTech cluster")
parser.add_argument("--scarb_cluster", action="store_true", help="Whether running on SCARB cluster")
parser.add_argument(
    "--model",
    type=str,
    default="100_genomes",
    choices=["100_genomes", "200_genomes", "400_genomes", "all_genomes"],
    help="Model variant to load (default: 100_genomes)",
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
    help="Sliding window stride in amino acids/codons for long sequences (default: 70, overlap=30 codons)",
)
parser.add_argument(
    "--input_format",
    type=str,
    default="csv",
    choices=["csv", "fasta"],
    help="Input file format: 'csv' (default) reads .csv.gz files, 'fasta' reads .fasta.gz files",
)
parser.add_argument(
    "--ancient_damage",
    action="store_true",
    help="Predict on ancient DNA samples (reads from fasta_ancient_damage/ subdirectory)",
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
    base_data_path = "/home/projects/DeepCDStmp/data/processed_data"
    input_data_dir_path = f"{base_data_path}/model_data/{model_dir_path_suffix}"
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
    # Local development
    base_data_path = "../../../data/processed_data"
    input_data_dir_path = f"{base_data_path}/model_data/shared_crf/{model_dir_path_suffix}"
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device_type = device.type
    num_workers_cpu = 4 if device_type == "cuda" else 0  # MPS can have issues with multiprocessing
    pin_memory = device_type == "cuda"

print(f"Device: {device}", flush=True)
print(f"DataLoader workers: {num_workers_cpu}, pin_memory: {pin_memory}")

model_name_ckpt = f"full_model_{args.model}_seed_42_trained_final.pth"

esm2_model_name = "facebook/esm2_t6_8M_UR50D"
esm2_model_abbr = esm2_model_name.split("/")[-1].split("_UR")[0]

test_samples_file = open(f"{base_data_path}/genome_partitions/test_partition_accessions.txt", "r")
test_samples = [line.strip() for line in test_samples_file.readlines()]
test_samples_file.close()


################################################################################################################################
################################################Helper Functions################################################################
################################################################################################################################

def clear_memory(sync=False):
    """Memory clean up function."""
    if torch.cuda.is_available():
        if sync:
            torch.cuda.synchronize()  # Wait for all GPU ops to complete
        torch.cuda.empty_cache()
    gc.collect()


_cached_tokenizer = None

def get_tokenizer():
    """Get cached tokenizer instance."""
    global _cached_tokenizer
    if _cached_tokenizer is None:
        _cached_tokenizer = AutoTokenizer.from_pretrained(
            "facebook/esm2_t6_8M_UR50D",
            do_lower_case=False,
        )
    return _cached_tokenizer


def parse_fasta_gz_to_df(fasta_gz_path):
    """
    Parse a fasta.gz file into a DataFrame for model inference.

    Header format: >read_name|strand|contig|cds_coords|seq_errors

    Returns:
        DataFrame with columns: read_name, read, cds_coords, indel_positions
    """
    rows = []
    with gzip.open(fasta_gz_path, 'rt') as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header is not None:
                    parts = header.split('|')
                    rows.append({
                        'read_name': parts[0],
                        'read': ''.join(seq_lines),
                        'cds_coords': parts[3] if len(parts) > 3 else '[]',
                        'indel_positions': parts[4] if len(parts) > 4 else 'None',
                    })
                header = line[1:]
                seq_lines = []
            elif line:
                seq_lines.append(line)
        if header is not None:
            parts = header.split('|')
            rows.append({
                'read_name': parts[0],
                'read': ''.join(seq_lines),
                'cds_coords': parts[3] if len(parts) > 3 else '[]',
                'indel_positions': parts[4] if len(parts) > 4 else 'None',
            })
    return pd.DataFrame(rows)


def load_and_process_data(test_sample, data_dir, batch_size, max_aa_len,
                          num_workers_cpu=num_workers_cpu, pin_memory=pin_memory):
    """
    Load and process test data for inference.

    Args:
        test_sample (str): Identifier for the test sample to load.
        data_dir (str): Directory name for the data type.
        batch_size (int): Batch size for DataLoader.
        max_aa_len (int): Maximum amino acid sequence length.
        num_workers_cpu (int): Number of CPU workers for data loading.
        pin_memory (bool): Whether to pin memory for DataLoader.

    Returns:
        DataLoader: DataLoader for the test data.
    """
    # Load data
    if args.ancient_damage:
        test_set = parse_fasta_gz_to_df(
            f"{base_data_path}/reads_processed/test/{data_dir}/fasta_ancient_damage/{test_sample}_ancient.fasta.gz"
        )
    elif args.input_format == "fasta":
        test_set = parse_fasta_gz_to_df(
            f"{base_data_path}/reads_processed/test/{data_dir}/fasta/{test_sample}.fasta.gz"
        )
    else:
        test_set = pd.read_csv(
            f"{base_data_path}/reads_processed/test/{data_dir}/csv/{test_sample}.csv.gz",
            index_col=None,
            compression="gzip"
        )

    print("Data samples: ", test_set.shape[0])

    # Use cached tokenizer
    tokenizer = get_tokenizer()

    # Process data using imported encode_data function
    test_data = encode_data(test_set, max_aa_len, tokenizer)

    # Clear intermediate data to save memory
    del test_set

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_cpu,
        pin_memory=pin_memory
    )

    return test_loader


def get_actual_sequence_length(input_ids, eos_token_id=2):
    """Find actual sequence length by locating EOS token."""
    actual_lengths = []
    for seq in input_ids:
        # Find EOS token position
        eos_positions = (seq == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            # Actual length is EOS position - 1 (excluding CLS)
            actual_length = eos_positions[0].item() - 1
        else:
            # If no EOS found, use full sequence minus CLS
            actual_length = len(seq) - 1
        actual_lengths.append(max(1, actual_length))  # Ensure at least 1
    return actual_lengths


def trim_predictions_by_eos(predictions, input_ids):
    """Trim predictions to actual sequence length based on EOS token."""
    actual_lengths = get_actual_sequence_length(input_ids, eos_token_id=2)
    trimmed_predictions = []
    for pred_seq, length in zip(predictions, actual_lengths):
        trimmed_pred = pred_seq[:length]
        trimmed_predictions.append(trimmed_pred)
    return trimmed_predictions


################################################################################################################################
################################################CDS Coordinate Extraction#######################################################
################################################################################################################################

@dataclass
class CDSSegment:
    start: int
    end: int
    frame: int
    start_type: str  # 'start_codon', 'indel_start', 'internal'
    end_type: str    # 'stop_codon', 'indel_stop', 'internal'
    group_id: Optional[str] = None  # Links related fragments
    indel_type: Optional[str] = None  # 'insertion', 'deletion', None


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
    Enhanced function to get predicted CDS coordinates with frameshift handling and uncertainty detection.

    Parameters:
        labels_rf0, labels_rf1, labels_rf2: prediction arrays for each reading frame

    Returns:
        segments: List of CDSSegment objects
        uncertain_regions: List of UncertainRegion objects
        transitions_info: List of Transition objects
        transition_positions: Dict of transition types and positions
    """
    # Initialize
    uncertain_regions = []
    transition_positions = {
        'start_codon': [],
        'stop_codon': [],
        'indel_start': [],
        'indel_stop': []
    }
    all_cds_fragments = []
    transitions_info = []

    # Get CDS fragment coordinates for each frame
    for rf, labels in enumerate([labels_rf0, labels_rf1, labels_rf2]):
        labels = np.array(labels)
        frame_segments, start_stop_codon_transitions = _extract_segments_from_frame(labels, rf, transition_positions)
        all_cds_fragments.extend(frame_segments)
        transitions_info.extend(start_stop_codon_transitions)

    # Sort CDS fragments by start position
    all_cds_fragments.sort(key=lambda x: x.start)

    # Connect fragments interrupted by frameshifts
    connected_segments = _connect_frameshift_segments(all_cds_fragments)

    # Create uncertain regions and trim segments for connected groups
    uncertain_regions, transitions_info = _create_uncertain_regions_from_groups(connected_segments, transitions_info)

    # Sort final segments
    connected_segments.sort(key=lambda x: x.start)

    # Sort start and stop codon positions
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
        nt_pos = i * 3 + rf + 1  # Convert to nucleotide position (1-indexed)

        if label in [1, 2, 4]:  # Start of CDS or inside CDS
            if not in_cds:
                in_cds = True
                start = nt_pos
                if label == 2:
                    start_type = 'start_codon'
                    transition_positions['start_codon'].append(nt_pos)
                    start_stop_codon_transition = Transition(
                        type="start_codon",
                        start_position=nt_pos,
                        end_position=nt_pos + 2,
                        frame=rf
                    )
                    start_stop_codon_transitions.append(start_stop_codon_transition)
                elif label == 4:
                    start_type = 'indel_start'
                    transition_positions['indel_start'].append(nt_pos)
                else:  # label == 1, coding but no explicit start
                    start_type = 'internal_region'

        elif label in [3, 5, 0]:  # End of CDS or non-coding
            if in_cds:
                if label == 3:
                    end_type = 'stop_codon'
                    end = nt_pos + 2  # Include stop codon
                    transition_positions['stop_codon'].append(end)
                    start_stop_codon_transition = Transition(
                        type="stop_codon",
                        start_position=nt_pos,
                        end_position=end,
                        frame=rf
                    )
                    start_stop_codon_transitions.append(start_stop_codon_transition)
                elif label == 5:
                    end_type = 'indel_stop'
                    end = nt_pos + 2
                    transition_positions['indel_stop'].append(end)
                else:  # label == 0, transition to non-coding
                    end_type = 'internal_region'
                    end = nt_pos - 1

                segment = CDSSegment(
                    start=start,
                    end=end,
                    frame=rf,
                    start_type=start_type,
                    end_type=end_type
                )
                segments.append(segment)

                in_cds = False
                start = None
                start_type = None

    # Handle case where CDS extends to end of sequence
    if in_cds:
        end = len(labels) * 3 + rf
        segment = CDSSegment(
            start=start,
            end=end,
            frame=rf,
            start_type=start_type,
            end_type='internal_region'
        )
        segments.append(segment)

    return segments, start_stop_codon_transitions


def _create_uncertain_regions_from_groups(segments, transitions):
    """Create uncertain regions between connected frameshift segments and trim overlapping parts."""
    uncertain_regions = []

    # Group segments by group_id
    groups = defaultdict(list)
    for segment in segments:
        if segment.group_id:
            groups[segment.group_id].append(segment)

    # Process each group to create uncertain regions and trim segments
    for group_id, group_segments in groups.items():
        if len(group_segments) < 2:
            continue

        # Sort segments in the group by start position
        group_segments.sort(key=lambda x: x.start)

        # Process consecutive pairs of segments in the group
        for i in range(len(group_segments) - 1):
            seg1 = group_segments[i]
            seg2 = group_segments[i + 1]

            # Determine the uncertain region between these two segments
            if seg1.end >= seg2.start:
                # Overlapping case - need to trim at codon boundaries
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
                        uncertain_region = UncertainRegion(
                            start=uncertain_start,
                            end=uncertain_end,
                            overlapping_frames=[seg1.frame, seg2.frame],
                            reason=f"Frameshift overlap between RF{seg1.frame} and RF{seg2.frame}"
                        )
                        uncertain_regions.append(uncertain_region)
                    elif uncertain_end == uncertain_start:
                        transition = Transition(
                            type="insertion",
                            start_position=uncertain_start,
                            end_position=uncertain_end,
                            frame=seg1.frame
                        )
                        transitions.append(transition)
            else:
                # Non-overlapping case - gap between segments
                gap_start = seg1.end + 1
                gap_end = seg2.start - 1

                if gap_end > gap_start:
                    uncertain_region = UncertainRegion(
                        start=gap_start,
                        end=gap_end,
                        overlapping_frames=[seg1.frame, seg2.frame],
                        reason=f"Frameshift gap between RF{seg1.frame} and RF{seg2.frame}"
                    )
                    uncertain_regions.append(uncertain_region)
                elif gap_end == gap_start:
                    transition = Transition(
                        type="insertion",
                        start_position=gap_start,
                        end_position=gap_end,
                        frame=seg1.frame
                    )
                    transitions.append(transition)

    return uncertain_regions, transitions


def detect_indel_type(from_frame, to_frame):
    """
    Detect indel type based on reading frame transition.

    Insertions cause forward jumps: 0->1, 1->2, 2->0
    Deletions cause backward jumps: 0->2, 1->0, 2->1
    """
    if from_frame == to_frame:
        return None

    forward_jumps = {(0, 1), (1, 2), (2, 0)}
    backward_jumps = {(0, 2), (1, 0), (2, 1)}

    transition = (from_frame, to_frame)

    if transition in forward_jumps:
        return 'insertion'
    elif transition in backward_jumps:
        return 'deletion'
    else:
        return 'complex'


def _connect_frameshift_segments(segments):
    """Attempt to connect segments that might be part of the same CDS interrupted by frameshifts."""
    connected_segments = []
    used_segments = set()
    group_counter = 1

    for i, segment in enumerate(segments):
        if i in used_segments:
            continue

        current_group = [segment]
        used_segments.add(i)

        # Look for segments that could be connected
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


def write_enhanced_gff(segments, uncertain_regions, transitions_info, read_name, cds_coords, seq_errors, outfile_gff):
    """Write segments and uncertain regions to GFF file with enhanced annotations."""
    counter_cds_frags_interrupted = {}

    # Write CDS segments
    for i, segment in enumerate(segments):
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

        attributes.append(f"ref={cds_coords}")
        attributes.append(f"seq_errors={seq_errors}")

        attr_string = ";".join(attributes)

        outfile_gff.write(
            f"{read_name}\tFragmentPredictor\tCDS\t{segment.start}\t{segment.end}\t"
            f".\t+\t{segment.frame}\t{attr_string}\n"
        )

    # Write start, stop and insertion positions as separate features
    for i, transition in enumerate(transitions_info):
        attributes = [f"ID={transition.type}_{read_name}_{i}"]
        attr_string = ";".join(attributes)

        outfile_gff.write(
            f"{read_name}\tFragmentPredictor\t{transition.type}\t{transition.start_position}\t{transition.end_position}\t"
            f".\t+\t.\t{attr_string}\n"
        )

    # Write uncertain regions as separate features
    for i, region in enumerate(uncertain_regions):
        attributes = []
        attributes.append(f"Note=Uncertain region: {region.reason}")
        attributes.append(f"overlapping_frames={','.join(map(str, region.overlapping_frames))}")

        involved_groups = set()
        for segment in segments:
            if (segment.group_id and
                not (segment.end < region.start or segment.start > region.end)):
                involved_groups.add(segment.group_id)

        if involved_groups:
            attributes.append(f"involved_groups={','.join(involved_groups)}")

        attr_string = ";".join(attributes)

        outfile_gff.write(
            f"{read_name}\tFragmentPredictor\tuncertain_region\t{region.start}\t{region.end}\t"
            f".\t+\t.\t{attr_string}\n"
        )


def process_predictions_enhanced(predictions_rf0, predictions_rf1, predictions_rf2,
                                  read_names, cds_coords, seq_errors, outfile_gff, batch_size):
    """Process and write predictions to GFF file."""
    for i in range(min(batch_size, len(cds_coords))):
        segments, uncertain_regions, transitions_info, transition_positions = get_cds_coords(
            predictions_rf0[i], predictions_rf1[i], predictions_rf2[i]
        )

        write_enhanced_gff(
            segments, uncertain_regions, transitions_info,
            read_names[i], cds_coords[i], seq_errors[i], outfile_gff
        )


################################################################################################################################
################################################Main Prediction Loop############################################################
################################################################################################################################

def run_model_predictions(data_dir, model, mapping_dict_to_class, max_aa_len,
                          test_samples=test_samples, batch_size=256, use_half_precision=True):
    """
    Run model predictions with optimized inference settings.

    Args:
        data_dir: Directory containing test data
        model: The trained model
        mapping_dict_to_class: Label mapping dictionary
        max_aa_len: Maximum amino acid sequence length
        test_samples: List of test sample identifiers
        batch_size: Batch size for inference
        use_half_precision: Whether to use FP16 inference (faster on MPS/CUDA)
    """
    # Enable half precision for faster inference on GPU/MPS
    if use_half_precision and device_type in ("cuda", "mps"):
        model = model.half()
        dtype = torch.float16
        print(f"Using half precision (FP16) inference on {device_type}")
    else:
        dtype = torch.float32

    for test_sample in tqdm(test_samples, desc="Processing samples"):
        test_loader = load_and_process_data(
            test_sample,
            data_dir=data_dir,
            batch_size=batch_size,
            max_aa_len=max_aa_len,
            num_workers_cpu=num_workers_cpu,
            pin_memory=pin_memory
        )

        data_dir_out = f"{data_dir}_ancient_damage" if args.ancient_damage else data_dir
        dir_path = f"{base_data_path}/predictions/raw_predictions/DeepCDS/{model_dir_path_suffix}/{data_dir_out}/{model_name_ckpt.split('.')[0]}/"
        os.makedirs(dir_path, exist_ok=True)
        outfile_gff = open(f"{dir_path}/predictions_{test_sample}.gff", "w")
        outfile_gff.write("##gff-version 3\n")

        # Use inference_mode for better performance than no_grad
        with torch.inference_mode():
            model.eval()
            for counter, batch in tqdm(enumerate(test_loader), total=len(test_loader)):

                # Move to device
                aa_encoding_rf0 = batch['aa_encodings_rf0']['input_ids'].to(device)
                rf0_attention_mask = batch['aa_encodings_rf0']['attention_mask'].to(device)
                aa_encoding_rf1 = batch['aa_encodings_rf1']['input_ids'].to(device)
                rf1_attention_mask = batch['aa_encodings_rf1']['attention_mask'].to(device)
                aa_encoding_rf2 = batch['aa_encodings_rf2']['input_ids'].to(device)
                rf2_attention_mask = batch['aa_encodings_rf2']['attention_mask'].to(device)

                cds_coords = batch['cds_coords']
                read_names = batch['read_name']
                seq_errors = batch['seq_errors']

                # Convert float tensors to half precision if enabled
                nt_encoding_rf0 = batch['nt_encodings_rf0'].to(device, dtype=dtype)
                nt_encoding_rf1 = batch['nt_encodings_rf1'].to(device, dtype=dtype)
                nt_encoding_rf2 = batch['nt_encodings_rf2'].to(device, dtype=dtype)

                # Predict with shared CRF
                outputs = model(
                    nt_encoding_rf0, aa_encoding_rf0, rf0_attention_mask,
                    nt_encoding_rf1, aa_encoding_rf1, rf1_attention_mask,
                    nt_encoding_rf2, aa_encoding_rf2, rf2_attention_mask
                )

                predictions_encoded = outputs["predictions"]

                # Decode predictions and split into rf0/rf1/rf2
                preds_rf0, preds_rf1, preds_rf2 = [], [], []

                for preds_sample in predictions_encoded:
                    preds = [mapping_dict_to_class[p] for p in preds_sample]
                    preds_rf0.append([rf[0] for rf in preds])
                    preds_rf1.append([rf[1] for rf in preds])
                    preds_rf2.append([rf[2] for rf in preds])

                preds_rf0 = trim_predictions_by_eos(preds_rf0, aa_encoding_rf0)
                preds_rf1 = trim_predictions_by_eos(preds_rf1, aa_encoding_rf1)
                preds_rf2 = trim_predictions_by_eos(preds_rf2, aa_encoding_rf2)

                process_predictions_enhanced(
                    preds_rf0, preds_rf1, preds_rf2,
                    read_names, cds_coords, seq_errors, outfile_gff, batch_size
                )

                # Clean up batch tensors to free GPU memory
                del aa_encoding_rf0, aa_encoding_rf1, aa_encoding_rf2
                del rf0_attention_mask, rf1_attention_mask, rf2_attention_mask
                del nt_encoding_rf0, nt_encoding_rf1, nt_encoding_rf2
                del outputs, predictions_encoded
                del preds_rf0, preds_rf1, preds_rf2

                # Periodic memory cleanup every 50 batches
                if (counter + 1) % 50 == 0:
                    clear_memory()

        outfile_gff.close()

        # Clean up after each sample
        del test_loader
        clear_memory(sync=True)

    # Convert model back to float32 if needed for further use
    if use_half_precision and device_type in ("cuda", "mps"):
        model = model.float()


def run_sliding_window_predictions(data_dir, model, mapping_dict_to_class, seq_len,
                                    test_samples=test_samples, batch_size=256,
                                    stride_aa=70, use_half_precision=True):
    """
    Run model predictions using sliding window inference for long sequences.

    Sequences are split into overlapping windows matching the training length (300 nt),
    logits are averaged in overlapping regions, and CRF decodes the full merged sequence.

    Args:
        data_dir: Directory containing test data
        model: The trained model
        mapping_dict_to_class: Label mapping dictionary
        seq_len: Nucleotide sequence length for this dataset
        test_samples: List of test sample identifiers
        batch_size: Base batch size for inference
        stride_aa: Sliding window stride in amino acids/codons
        use_half_precision: Whether to use FP16 inference
    """
    if use_half_precision and device_type in ("cuda", "mps"):
        model = model.half()
        dtype = torch.float16
        print(f"Using half precision (FP16) inference on {device_type}")
    else:
        dtype = torch.float32

    tokenizer = get_tokenizer()

    for test_sample in tqdm(test_samples, desc="Processing samples"):
        if args.ancient_damage:
            test_df = parse_fasta_gz_to_df(
                f"{base_data_path}/reads_processed/test/{data_dir}/fasta_ancient_damage/{test_sample}_ancient.fasta.gz"
            )
        elif args.input_format == "fasta":
            test_df = parse_fasta_gz_to_df(
                f"{base_data_path}/reads_processed/test/{data_dir}/fasta/{test_sample}.fasta.gz"
            )
        else:
            test_df = pd.read_csv(
                f"{base_data_path}/reads_processed/test/{data_dir}/csv/{test_sample}.csv.gz",
                index_col=None,
                compression="gzip"
            )

        print(f"Data samples: {test_df.shape[0]}")

        data_dir_out = f"{data_dir}_ancient_damage" if args.ancient_damage else data_dir
        dir_path = f"{base_data_path}/predictions/raw_predictions/DeepCDS/{model_dir_path_suffix}/{data_dir_out}/{model_name_ckpt.split('.')[0]}/"
        os.makedirs(dir_path, exist_ok=True)
        outfile_gff = open(f"{dir_path}/predictions_{test_sample}.gff", "w")
        outfile_gff.write("##gff-version 3\n")

        with torch.inference_mode():
            model.eval()
            chunk_counter = 0

            for (preds_rf0, preds_rf1, preds_rf2,
                 read_names, cds_coords, seq_errors, chunk_size) in sliding_window_inference(
                    model=model,
                    sequences_df=test_df,
                    seq_len=seq_len,
                    mapping_dict_to_class=mapping_dict_to_class,
                    tokenizer=tokenizer,
                    device=device,
                    dtype=dtype,
                    batch_size=batch_size,
                    stride_aa=stride_aa,
                    num_workers_cpu=num_workers_cpu,
                    pin_memory=pin_memory,
            ):
                process_predictions_enhanced(
                    preds_rf0, preds_rf1, preds_rf2,
                    read_names, cds_coords, seq_errors, outfile_gff, chunk_size
                )

                chunk_counter += 1
                if chunk_counter % 10 == 0:
                    clear_memory()

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

    if args.ancient_damage:
        print("Ancient DNA damage patterns will be included in the input data (simulated C->T and G->A substitutions at read ends).")
        data_dirs = [
                    "without_errors_60bp",
                    "without_errors_75bp",
                    "without_errors_100bp",
                    "without_errors_150bp",
                    "without_errors_300bp"]
        model, mapping_dict_to_class = load_model(
            model_name_ckpt,
            input_data_dir_path,
            device=device,
            esm2_model=esm2_model_name,
            label_classes=label_classes
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
                max_aa_len = int(np.ceil(seq_len / 3)) + 5  # add padding buffer
                run_model_predictions(data_dir, model, mapping_dict_to_class, max_aa_len, batch_size=args.batch_size)

        

    else:
        # Define data directories based on error type
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
                # For smaller models, only predict on 300bp dataset
                data_dirs = ["without_errors_300bp"]
                print(f"Note: Using only 300bp dataset for model '{args.model}' (use --model all_genomes for full evaluation)")

        elif args.error_type in ("indel_substitution", "substitution"):
            # Error profiles: low (5e-06i/0.004s), medium (1.25e-05i/0.01s), high (3.75e-05i/0.03s)
            error_profiles = [
                "with_errors_5e-06i_0.004s",
                "with_errors_1.25e-05i_0.01s",
                "with_errors_3.75e-05i_0.03s",
            ]
            if args.model == "all_genomes":
                read_lengths = ["60bp", "75bp", "100bp", "150bp", "300bp", "700bp", "1000bp"]
                data_dirs = [f"{profile}_{length}" for profile in error_profiles for length in read_lengths]
            else:
                # For smaller models, only predict on 300bp datasets
                data_dirs = [f"{profile}_300bp" for profile in error_profiles]
                print(f"Note: Using only 300bp datasets for model '{args.model}' (use --model all_genomes for full evaluation)")

        else:
            raise ValueError(f"Unknown error_type: '{args.error_type}'")

        model, mapping_dict_to_class = load_model(
            model_name_ckpt,
            input_data_dir_path,
            device=device,
            esm2_model=esm2_model_name,
            label_classes=label_classes
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
                max_aa_len = int(np.ceil(seq_len / 3)) + 5  # add padding buffer
                run_model_predictions(data_dir, model, mapping_dict_to_class, max_aa_len, batch_size=args.batch_size)
