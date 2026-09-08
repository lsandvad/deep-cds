"""
DeepCDS Prediction Script - CAMI metagenome test set

Runs CDS predictions on the CAMI metagenome test set using the three trained
DeepCDS model variants: N (no errors), S (substitution errors), S+I
(substitution + indel errors).

The metagenome fasta is far larger (tens of millions of reads) than the
per-genome test sets the original predict_with_DeepCDS.py was built for, so
reads are streamed and processed in chunks to bound memory usage instead of
loading the whole file into a single DataFrame.
"""

import argparse
import gc
import glob
import gzip
import logging
import os
import resource
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src_dev.modeling import (
    TRAINED_WINDOW_SIZE_AA,
    build_label_lut,
    codon_one_hot_from_codes,
    encode_reads_fast,
    load_model,
    sliding_window_inference,
    viterbi_decode_fast,
)

logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

pd.options.mode.chained_assignment = None  # Suppress the warning globally
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

################################################################################################################################
################################################Argument Parser#################################################################
################################################################################################################################

parser = argparse.ArgumentParser(description="Run DeepCDS (N / S / S+I) predictions on the CAMI metagenome test set")
parser.add_argument("--gpu", type=int, default=0, help="GPU number to use (default: 0)")
parser.add_argument(
    "--project_path", type=str, default=None,
    help="Explicit FragmentPredictor project root (the directory containing data/processed_data). "
         "Overrides --scarb_cluster / --healthtech_cluster, and can also be set via the "
         "DEEPCDS_PROJECT_PATH environment variable. Use this so the .sh wrapper's sshfs mount "
         "point and the path this script reads are one value and cannot drift apart.",
)
parser.add_argument("--scarb_cluster", action="store_true", help="Use SCARB cluster path (/tmp/nrt204/FragmentPredictor)")
parser.add_argument("--healthtech_cluster", action="store_true",
                    help="Use HealthTech cluster path "
                         "(/net/well/pool/projects2/lisani/DeepCDS/FragmentPredictor) - the .sh wrappers "
                         "must sshfs-mount ERDA at exactly that path")
parser.add_argument(
    "--model",
    type=str,
    default="all_genomes",
    choices=["100_genomes", "200_genomes", "400_genomes", "all_genomes"],
    help="Model variant to load (default: all_genomes)",
)
parser.add_argument(
    "--batch_size", type=int, default=1024,
    help="Batch size for inference (default: 1024). Each batch runs all three reading "
         "frames as one 3*batch_size forward pass; at 1024 the 8M model peaks around "
         "2GB of GPU memory, so raise it further if the card allows.",
)
parser.add_argument(
    "--chunk_size",
    type=int,
    default=100_000,
    help="Number of reads to stream and process at a time, to bound memory usage on the large metagenome file (default: 100000)",
)
parser.add_argument(
    "--stride_aa",
    type=int,
    default=70,
    help="Sliding window stride in amino acids/codons for reads longer than the trained window (default: 70)",
)
parser.add_argument(
    "--esm_model",
    type=str,
    default="8M",
    choices=["8M", "35M", "150M", "650M"],
    help="ESM-2 model size to use (default: 8M)",
)
parser.add_argument(
    "--error_type",
    type=str,
    default=None,
    choices=["none", "substitution", "indel_substitution"],
    help="Run only this DeepCDS variant (none=N, substitution=S, indel_substitution=S+I). "
         "Default: run all three variants in sequence.",
)
parser.add_argument(
    "--mem_log_interval",
    type=int,
    default=200,
    help="Print host/GPU memory usage every N batches, to the same stdout the .sh wrapper "
         "redirects to a .txt log. Set to 0 to disable (default: 200)",
)
parser.add_argument(
    "--amp",
    type=str,
    default="off",
    choices=["off", "bf16", "fp16"],
    help="Autocast dtype for the model forward pass. Keep 'off' (default). bf16 was "
         "measured to degrade predictions badly on genomes outside the training "
         "distribution - F1 on virus genomes fell from ~0.8 to ~0.5 - because bf16 keeps "
         "only ~7 mantissa bits and out-of-distribution reads have narrow CRF score "
         "margins that rounding then flips. Do not enable without re-validating on "
         "out-of-scope genomes.",
)
parser.add_argument(
    "--tf32",
    action="store_true",
    help="Allow TF32 for fp32 matmuls on Ampere+ GPUs. OFF by default, matching PyTorch's "
         "own default: TF32 keeps 10 mantissa bits against fp32's 23, which is the same "
         "kind of precision loss that made --amp bf16 unusable here, just milder. It is "
         "faster, but predictions are no longer bit-identical to an fp32 run, so validate "
         "on out-of-scope genomes before trusting it.",
)
parser.add_argument(
    "--aa_pad_buffer",
    type=int,
    default=5,
    help="Extra codon positions padded onto the per-chunk window beyond ceil(max_read_len/3) "
         "(default: 5, as in all previous runs - keep it to stay comparable with them). "
         "Lowering it is ~2%% faster per position dropped but DOES change predictions: the "
         "buffer keeps each frame's EOS position inside the trimmed window, where it still "
         "acts as an attention key for the real codon positions.",
)
parser.add_argument(
    "--count_reads",
    dest="count_reads",
    action="store_true",
    default=True,
    help="Count reads up front (a full extra decompression pass) so the progress bar has an ETA. "
         "The count is cached next to the predictions, so only the first run pays for it.",
)
parser.add_argument(
    "--no_count_reads",
    dest="count_reads",
    action="store_false",
    help="Skip the read-counting pass entirely; the progress bar then shows a rate but no ETA.",
)
args = parser.parse_args()

if args.scarb_cluster and args.healthtech_cluster:
    raise ValueError("Pass only one of --scarb_cluster or --healthtech_cluster.")

################################################################################################################################
################################################Device Configuration############################################################
################################################################################################################################

if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
device_type = device.type
print(f"Running on device: {device} ({device_type})", flush=True)

num_workers_cpu = 4 if device_type == "cuda" else 0  # MPS can have issues with multiprocessing
pin_memory = device_type == "cuda"

# An explicit path always wins, so a wrapper can hand over the exact directory it
# mounted instead of both sides hard-coding a path that has to be kept in sync.
explicit_project_path = args.project_path or os.environ.get("DEEPCDS_PROJECT_PATH")

if explicit_project_path:
    project_path = explicit_project_path.rstrip("/")
elif args.healthtech_cluster:
    # sshfs-mounted from the same ERDA remote (line.s.nielsen@bio.ku.dk@io.erda.dk:FragmentPredictor)
    # as the SCARB mount, just at a different local mount point -> identical structure under ERDA_ROOT.
    project_path = "/net/well/pool/projects2/lisani/DeepCDS/FragmentPredictor1"
elif args.scarb_cluster:
    project_path = "/tmp/nrt204/FragmentPredictor"
else:
    project_path = "../../.."

base_data_path = f"{project_path}/data/processed_data"
model_data_subpath = "model_data/shared_crf"  # same layout under ERDA_ROOT regardless of cluster mount point

data_dir = "CAMI_metagenome"
input_dir = f"{base_data_path}/reads_processed/test/{data_dir}"

if args.esm_model == "8M":
    esm2_model_name = "facebook/esm2_t6_8M_UR50D"
elif args.esm_model == "35M":
    esm2_model_name = "facebook/esm2_t12_35M_UR50D"
elif args.esm_model == "150M":
    esm2_model_name = "facebook/esm2_t30_150M_UR50D"
elif args.esm_model == "650M":
    esm2_model_name = "facebook/esm2_t33_650M_UR50D"

model_name_ckpt = f"full_model_{args.model}_seed_42_trained_final_{args.esm_model}_no_dropout.pth"

# DeepCDS N / S / S+I variants
error_type_configs = {
    "none": {"label": "N", "model_dir_path_suffix": "model_without_errors", "label_classes": 4},
    "substitution": {"label": "S", "model_dir_path_suffix": "model_with_substitution_errors", "label_classes": 4},
    "indel_substitution": {"label": "S+I", "model_dir_path_suffix": "model_with_errors", "label_classes": 6},
}

if args.error_type is not None:
    error_type_configs = {args.error_type: error_type_configs[args.error_type]}

################################################################################################################################
################################################Helper Functions################################################################
################################################################################################################################

def clear_memory(sync=False):
    """Memory clean up function.

    Deliberately *not* called per chunk: a full ``gc.collect()`` plus
    ``empty_cache()` a few hundred times over a 33M-read run costs real time and
    buys nothing, because the per-chunk arrays are plain NumPy and are freed by
    refcounting the moment the chunk goes out of scope.
    """
    if torch.cuda.is_available():
        if sync:
            torch.cuda.synchronize()  # Wait for all GPU ops to complete
        torch.cuda.empty_cache()
    gc.collect()


def log_memory_usage(step):
    """Print host RSS (peak, via stdlib resource - no extra dependency) and GPU memory
    stats for the current device, so they land in the same .txt log the .sh wrapper
    already redirects stdout to."""
    host_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)  # ru_maxrss is in KB on Linux
    if torch.cuda.is_available():
        gpu_alloc_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        gpu_reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
        gpu_peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"[mem] step={step} host_rss={host_rss_gb:.2f}GB "
              f"gpu_alloc={gpu_alloc_gb:.2f}GB gpu_reserved={gpu_reserved_gb:.2f}GB gpu_peak={gpu_peak_gb:.2f}GB",
              flush=True)
    else:
        print(f"[mem] step={step} host_rss={host_rss_gb:.2f}GB", flush=True)


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


def count_fasta_records(fasta_gz_path, cache_dir=None):
    """Count fasta records via a fast `zcat | wc -l` pass (2 lines per record).

    The count is only used to give the progress bar an ETA, but the pass costs a
    full decompression of a ~600MB gzip - and the three DeepCDS variants are
    normally submitted as three separate jobs over the same file. Caching it in
    `cache_dir` means only the first of them pays.
    """
    cache_path = None
    if cache_dir:
        stat = os.stat(fasta_gz_path)
        cache_path = os.path.join(
            cache_dir, f".{os.path.basename(fasta_gz_path)}.{stat.st_size}.nreads"
        )
        try:
            with open(cache_path) as cache_file:
                return int(cache_file.read().strip())
        except (OSError, ValueError):
            pass

    result = subprocess.run(
        f"zcat {fasta_gz_path} | wc -l",
        shell=True, capture_output=True, text=True, check=True,
    )
    count = int(result.stdout.strip()) // 2

    if cache_path:
        try:
            with open(cache_path, "w") as cache_file:
                cache_file.write(str(count))
        except OSError:
            pass  # a read-only mount just means we recount next time
    return count


def _open_fasta_gz(fasta_gz_path):
    """Open a fasta.gz for line iteration, decompressing in a separate process if possible.

    Piping through `zcat` moves inflate off the Python process entirely, so it
    overlaps with the GIL-bound parsing instead of competing with it. Falls back
    to the stdlib when zcat isn't on PATH.
    """
    zcat = shutil.which("zcat")
    if zcat is None:
        return gzip.open(fasta_gz_path, "rb"), None
    proc = subprocess.Popen([zcat, fasta_gz_path], stdout=subprocess.PIPE, bufsize=1 << 20)
    return proc.stdout, proc


def iter_fasta_gz_chunks(fasta_gz_path, chunk_size):
    """
    Stream a fasta.gz file, yielding chunks of up to `chunk_size` reads instead of
    loading the whole (tens-of-millions-of-reads) file at once.

    Yields plain column lists rather than a DataFrame: the fixed-window path feeds
    them straight into the vectorised encoder, and building 300+ DataFrames of
    100k dict rows is pure overhead (only the rarely-taken sliding-window path
    wants a DataFrame, and builds one on demand).

    Header format: >read_name|strand|contig|cds_coords|seq_errors|...

    Yields:
        dict: with 'read_name', 'read', 'cds_coords', 'indel_positions' lists and
        'max_read_len'.
    """
    read_names, reads, cds_coords, indel_positions = [], [], [], []
    max_read_len = 0
    header = None
    seq_lines = []

    def flush_record():
        nonlocal max_read_len
        parts = header.split('|')
        read = ''.join(seq_lines) if len(seq_lines) != 1 else seq_lines[0]
        read_names.append(parts[0])
        reads.append(read)
        cds_coords.append(parts[3] if len(parts) > 3 else '[]')
        indel_positions.append(parts[4] if len(parts) > 4 else 'None')
        if len(read) > max_read_len:
            max_read_len = len(read)

    def make_chunk():
        return {
            'read_name': read_names, 'read': reads,
            'cds_coords': cds_coords, 'indel_positions': indel_positions,
            'max_read_len': max_read_len,
        }

    stream, proc = _open_fasta_gz(fasta_gz_path)
    try:
        for raw in stream:
            line = raw.decode('ascii').strip()
            if not line:
                continue
            if line[0] == '>':
                if header is not None:
                    flush_record()
                    if len(reads) >= chunk_size:
                        yield make_chunk()
                        read_names, reads, cds_coords, indel_positions = [], [], [], []
                        max_read_len = 0
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            flush_record()
    finally:
        stream.close()
        if proc is not None:
            proc.wait()

    if reads:
        yield make_chunk()


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
    Get predicted CDS coordinates with frameshift handling and uncertainty detection.
    """
    uncertain_regions = []
    transition_positions = {
        'start_codon': [],
        'stop_codon': [],
        'indel_start': [],
        'indel_stop': []
    }
    all_cds_fragments = []
    transitions_info = []

    for rf, labels in enumerate([labels_rf0, labels_rf1, labels_rf2]):
        # labels arrive as plain Python int lists; _extract_segments_from_frame only
        # enumerates them, so converting to ndarray here just costs an allocation per
        # read per frame (~100M allocations over the metagenome) and makes the loop
        # below iterate NumPy scalars, which compare far slower than ints.
        frame_segments, start_stop_codon_transitions = _extract_segments_from_frame(labels, rf, transition_positions)
        all_cds_fragments.extend(frame_segments)
        transitions_info.extend(start_stop_codon_transitions)

    all_cds_fragments.sort(key=lambda x: x.start)

    connected_segments = _connect_frameshift_segments(all_cds_fragments)

    uncertain_regions, transitions_info = _create_uncertain_regions_from_groups(connected_segments, transitions_info)

    connected_segments.sort(key=lambda x: x.start)
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
                    start_stop_codon_transitions.append(Transition(
                        type="start_codon", start_position=nt_pos, end_position=nt_pos + 2, frame=rf
                    ))
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
                    start_stop_codon_transitions.append(Transition(
                        type="stop_codon", start_position=nt_pos, end_position=end, frame=rf
                    ))
                elif label == 5:
                    end_type = 'indel_stop'
                    end = nt_pos + 2
                    transition_positions['indel_stop'].append(end)
                else:  # label == 0, transition to non-coding
                    end_type = 'internal_region'
                    end = nt_pos - 1

                segments.append(CDSSegment(start=start, end=end, frame=rf, start_type=start_type, end_type=end_type))

                in_cds = False
                start = None
                start_type = None

    if in_cds:
        end = len(labels) * 3 + rf
        segments.append(CDSSegment(start=start, end=end, frame=rf, start_type=start_type, end_type='internal_region'))

    return segments, start_stop_codon_transitions


def _create_uncertain_regions_from_groups(segments, transitions):
    """Create uncertain regions between connected frameshift segments and trim overlapping parts."""
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
                            type="insertion", start_position=uncertain_start, end_position=uncertain_end, frame=seg1.frame
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
                        type="insertion", start_position=gap_start, end_position=gap_end, frame=seg1.frame
                    ))

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
    """Append the GFF lines for one read to `outfile_gff`.

    `outfile_gff` only needs an `.write(str)`; the caller passes a list-backed
    buffer so a whole batch is written to disk in one call rather than one
    syscall-per-line for ~33M reads.
    """
    counter_cds_frags_interrupted = {}

    for segment in segments:
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

        attr_string = ";".join(attributes)

        outfile_gff.write(
            f"{read_name}\tFragmentPredictor\tCDS\t{segment.start}\t{segment.end}\t"
            f".\t+\t{segment.frame}\t{attr_string}\n"
        )

    for i, transition in enumerate(transitions_info):
        attr_string = f"ID={transition.type}_{read_name}_{i}"
        outfile_gff.write(
            f"{read_name}\tFragmentPredictor\t{transition.type}\t{transition.start_position}\t{transition.end_position}\t"
            f".\t+\t.\t{attr_string}\n"
        )

    for region in uncertain_regions:
        attributes = [f"Note=Uncertain region: {region.reason}",
                      f"overlapping_frames={','.join(map(str, region.overlapping_frames))}"]

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


class GffLineBuffer:
    """Collects GFF lines for one batch so they reach the file in a single write().

    `write_enhanced_gff` emits a handful of short lines per read; sending each of
    those straight to the file object costs a Python-level buffered write per line
    across tens of millions of reads. Buffering per batch and joining once cuts
    that to one write per batch.
    """

    __slots__ = ("lines",)

    def __init__(self):
        self.lines = []

    def write(self, text):
        self.lines.append(text)

    def flush_to(self, outfile_gff):
        if self.lines:
            outfile_gff.write("".join(self.lines))
            self.lines.clear()


def process_predictions_enhanced(predictions_rf0, predictions_rf1, predictions_rf2,
                                  read_names, cds_coords, seq_errors, outfile_gff, batch_size):
    """Process and write predictions to GFF file."""
    buffer = GffLineBuffer()
    for i in range(min(batch_size, len(cds_coords))):
        segments, uncertain_regions, transitions_info, _ = get_cds_coords(
            predictions_rf0[i], predictions_rf1[i], predictions_rf2[i]
        )
        write_enhanced_gff(segments, uncertain_regions, transitions_info,
                            read_names[i], cds_coords[i], seq_errors[i], buffer)
    buffer.flush_to(outfile_gff)


################################################################################################################################
################################################Chunk Inference##################################################################
################################################################################################################################

def prepare_chunk_fixed(chunk, max_aa_len):
    """CPU-only prep for one fixed-window chunk.

    Replaces the old ``encode_data() + DataLoader`` pair. ``encode_reads_fast``
    produces exactly the same ``input_ids`` / ``attention_mask`` / codon one-hot
    contents (verified bit-for-bit against ``encode_data``), but as a handful of
    whole-chunk NumPy arrays built in vectorised passes, instead of running the
    ESM tokenizer - a pure-Python character-Trie walk - over 3 x chunk_size
    amino-acid strings and allocating two tensors per read per reading frame.

    Nucleotides are kept as one byte per base rather than an expanded (N, 12)
    float one-hot: the expansion is done on the GPU in
    ``codon_one_hot_from_codes``, which cuts both the host memory for a chunk and
    the per-batch host->device transfer by ~48x.

    Does no GPU work at all, so it is safe to run on the prefetch thread while the
    GPU is still busy with the previous chunk - see prefetch_chunks() below.
    """
    return encode_reads_fast(chunk["read"], max_aa_len, read_len=chunk["max_read_len"])


def run_prepared_chunk(model, label_lut, prepared, dtype, outfile_gff, batch_size,
                        amp_dtype=None, pbar=None, start_step=0, mem_log_interval=0):
    """Run the GPU inference loop over one already-encoded chunk.
    This is the part that must stay on the main thread/CUDA context.

    Returns:
        (n_reads, step): step is start_step + number of batches consumed, so the caller
        can keep a running batch counter across chunks for --mem_log_interval logging.
    """
    encoded = prepared["encoded"]
    meta = prepared["meta"]

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    nt_codes = encoded["nt_codes"]
    trim_lengths = encoded["trim_lengths"]

    read_names_all = meta["read_name"]
    cds_coords_all = meta["cds_coords"]
    seq_errors_all = meta["indel_positions"]

    n_reads = len(read_names_all)
    step = start_step
    autocast_enabled = amp_dtype is not None

    for begin in range(0, n_reads, batch_size):
        stop = min(begin + batch_size, n_reads)

        # Token ids and masks cross the bus as int16/int8 and are widened on the
        # device; nucleotides cross as one byte per base and are expanded into the
        # (N, 12) codon one-hot there, so the transfer is ~48x smaller than shipping
        # the float one-hot the old DataLoader collated on the host.
        aa_frames, mask_frames, nt_frames = [], [], []
        for rf in range(3):
            aa_frames.append(
                torch.from_numpy(input_ids[rf, begin:stop]).to(device, non_blocking=True).long()
            )
            mask_frames.append(
                torch.from_numpy(attention_mask[rf, begin:stop]).to(device, non_blocking=True).long()
            )
            codes = torch.from_numpy(nt_codes[rf, begin:stop]).to(device, non_blocking=True)
            nt_frames.append(codon_one_hot_from_codes(codes, dtype=dtype))

        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=autocast_enabled):
            # All three reading frames in one batched pass through the shared stack.
            logits, combined_mask = model.predict_logits(nt_frames, aa_frames, mask_frames)

        # Viterbi in fp32 regardless of autocast, so --amp only affects the emissions.
        tags, lengths = viterbi_decode_fast(model.CRF.crf, logits.float(), combined_mask.bool())

        # The single host sync of the whole batch (torchcrf's own backtrace does one
        # per token per sample, which is what made this loop GPU-latency bound).
        tags_np = tags.cpu().numpy()
        lengths_np = lengths.cpu().numpy()

        # (3, B, N) per-reading-frame labels in one fancy-index + one .tolist().
        rf_labels = np.ascontiguousarray(label_lut[tags_np].transpose(2, 0, 1))
        preds_rf0, preds_rf1, preds_rf2 = rf_labels.tolist()

        # Same trimming torchcrf's mask + trim_predictions_by_eos performed: the CRF
        # only decoded `lengths` positions, and each frame is further cut to its own
        # EOS position (precomputed by the encoder).
        batch_len = stop - begin
        trims = np.minimum(trim_lengths[:, begin:stop], lengths_np[None, :])
        for rf, preds in enumerate((preds_rf0, preds_rf1, preds_rf2)):
            trim_rf = trims[rf]
            for i in range(batch_len):
                preds[i] = preds[i][:trim_rf[i]]

        process_predictions_enhanced(
            preds_rf0, preds_rf1, preds_rf2,
            read_names_all[begin:stop], cds_coords_all[begin:stop], seq_errors_all[begin:stop],
            outfile_gff, batch_len,
        )
        if pbar is not None:
            pbar.update(batch_len)

        step += 1
        if mem_log_interval and step % mem_log_interval == 0:
            log_memory_usage(step)

    return n_reads, step


def run_chunk_sliding(model, mapping_dict_to_class, chunk_df, seq_len, tokenizer, batch_size, stride_aa, dtype, outfile_gff, pbar=None):
    """Run sliding-window inference (reads longer than the trained window) on one in-memory chunk of reads."""
    n_reads = 0
    for (preds_rf0, preds_rf1, preds_rf2, read_names, cds_coords, seq_errors, chunk_sz) in sliding_window_inference(
        model=model, sequences_df=chunk_df, seq_len=seq_len, mapping_dict_to_class=mapping_dict_to_class,
        tokenizer=tokenizer, device=device, dtype=dtype, batch_size=batch_size, stride_aa=stride_aa,
        num_workers_cpu=num_workers_cpu, pin_memory=pin_memory,
    ):
        process_predictions_enhanced(preds_rf0, preds_rf1, preds_rf2,
                                      read_names, cds_coords, seq_errors, outfile_gff, chunk_sz)
        n_reads += chunk_sz
        if pbar is not None:
            pbar.update(chunk_sz)
    return n_reads


def _prepare_chunk(chunk, trained_window_nt, aa_pad_buffer):
    """Background-thread task: decide fixed-vs-sliding routing for one chunk, and for
    the (common) fixed-window path, run the CPU-heavy encoding ahead of time.

    The sliding-window path isn't prepared here - sliding_window_inference does its
    own encoding internally per-window, so there's nothing to precompute for it
    without reaching into that module; it only gets the DataFrame it expects.
    """
    seq_len = chunk["max_read_len"]
    if seq_len > trained_window_nt:
        chunk_df = pd.DataFrame({k: chunk[k] for k in ("read_name", "read", "cds_coords", "indel_positions")})
        return {"kind": "sliding", "chunk_df": chunk_df, "seq_len": seq_len}

    # Codon positions per frame. The buffer is NOT free padding: it widens the window
    # enough that each frame's EOS token stays inside the trimmed attention mask, so it
    # participates in self-attention over the real codons. Shrinking it therefore changes
    # predictions, which is why the default stays at the 5 previous runs used.
    max_aa_len = -(-seq_len // 3) + aa_pad_buffer
    encoded = prepare_chunk_fixed(chunk, max_aa_len)
    return {"kind": "fixed", "encoded": encoded, "meta": chunk}


def prefetch_chunks(fasta_gz_path, chunk_size, trained_window_nt, aa_pad_buffer, executor):
    """Yields prepared chunk dicts (see _prepare_chunk), one chunk ahead of consumption:
    by the time this yields chunk N, chunk N+1's read-from-disk + encoding is already
    running on `executor`'s background thread, so that CPU-only work overlaps with
    chunk N's GPU time in the caller instead of happening serially after it.
    The encoder spends nearly all of its time inside NumPy, which releases the GIL,
    so the overlap is real rather than nominal.
    """
    chunk_iter = iter_fasta_gz_chunks(fasta_gz_path, chunk_size)

    def submit_next():
        try:
            chunk = next(chunk_iter)
        except StopIteration:
            return None
        return executor.submit(_prepare_chunk, chunk, trained_window_nt, aa_pad_buffer)

    next_future = submit_next()
    while next_future is not None:
        current_future = next_future
        next_future = submit_next()  # kick off N+1's prep before handing back chunk N
        yield current_future.result()


################################################################################################################################
################################################Main Entry Point################################################################
################################################################################################################################

if __name__ == "__main__":
    print(f"Model checkpoint: {model_name_ckpt}")
    print(f"Batch size: {args.batch_size}, chunk size: {args.chunk_size}, "
          f"amp: {args.amp}, tf32: {args.tf32}")

    fasta_gz_paths = sorted(glob.glob(f"{input_dir}/*_cds_labels.fasta.gz"))
    if not fasta_gz_paths:
        # On the sshfs-backed clusters an empty/absent input dir almost always means the
        # mount silently failed, so say that rather than only naming the missing glob.
        if not os.path.isdir(project_path):
            detail = f"project root {project_path} does not exist - is ERDA mounted there?"
        elif not os.path.isdir(input_dir):
            detail = (f"project root {project_path} exists but {input_dir} does not - "
                      "the mount may have failed or be pointing at the wrong remote.")
        else:
            detail = f"{input_dir} exists but contains no *_cds_labels.fasta.gz."
        raise FileNotFoundError(f"No *_cds_labels.fasta.gz files found. {detail}")

    trained_window_nt = TRAINED_WINDOW_SIZE_AA * 3  # trained window size in nucleotides
    tokenizer = get_tokenizer()  # only the sliding-window path still needs it
    dtype = torch.float32
    amp_dtype = {"off": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]

    if device_type == "cuda" and args.tf32:
        # Opt-in only. TF32 rounds fp32 matmul inputs to 10 mantissa bits, so it is a
        # precision trade-off rather than free speed - see --tf32. Left at PyTorch's own
        # default (False) otherwise, so the fp32 path stays a true fp32 path.
        torch.backends.cuda.matmul.allow_tf32 = True

    output_root = f"{base_data_path}/predictions/raw_predictions/DeepCDS"
    os.makedirs(output_root, exist_ok=True)

    # Count reads once per sample (reused across every error-type model) so the
    # per-read progress bar can show a real ETA instead of just a rate.
    sample_read_counts = {}
    if args.count_reads:
        print("Counting reads per sample...", flush=True)
        for gz_path in fasta_gz_paths:
            sample_read_counts[gz_path] = count_fasta_records(gz_path, cache_dir=output_root)
            print(f"  {os.path.basename(gz_path)}: {sample_read_counts[gz_path]} reads", flush=True)

    timing_log_path = f"{output_root}/{data_dir}_timing_log.tsv"
    with open(timing_log_path, "w") as timing_log:
        timing_log.write("sample\terror_type\tn_reads\tinference_time_s\treads_per_s\n")

    for error_type, cfg in error_type_configs.items():
        print(f"\n=== DeepCDS {cfg['label']} (error_type={error_type}) ===", flush=True)

        model_dir_path_suffix = cfg["model_dir_path_suffix"]
        input_data_dir_path = f"{base_data_path}/{model_data_subpath}/{model_dir_path_suffix}"

        model, mapping_dict_to_class = load_model(
            model_name_ckpt, input_data_dir_path, device=device,
            esm2_model=esm2_model_name, label_classes=cfg["label_classes"],
        )
        model.eval()
        # {encoded label -> (rf0, rf1, rf2)} as an array, so decoding a batch is one
        # fancy-index instead of a dict lookup per token.
        label_lut = build_label_lut(mapping_dict_to_class)

        for gz_path in fasta_gz_paths:
            sample = os.path.basename(gz_path).replace("_cds_labels.fasta.gz", "")
            print(f"{sample} (DeepCDS {cfg['label']})", flush=True)

            dir_path = f"{output_root}/{model_dir_path_suffix}/{data_dir}/{model_name_ckpt.split('.')[0]}/"
            os.makedirs(dir_path, exist_ok=True)
            outfile_gff = open(f"{dir_path}/predictions_{sample}.gff", "w", buffering=1 << 22)
            outfile_gff.write("##gff-version 3\n")

            sample_start = time.time()
            n_reads = 0
            step = 0  # running batch counter, for --mem_log_interval

            with torch.inference_mode(), tqdm(total=sample_read_counts.get(gz_path), unit="seq",
                                               desc=f"{sample} (DeepCDS {cfg['label']})") as pbar, \
                 ThreadPoolExecutor(max_workers=1) as prefetch_executor:
                for prepared in prefetch_chunks(gz_path, args.chunk_size, trained_window_nt,
                                                 args.aa_pad_buffer, prefetch_executor):
                    if prepared["kind"] == "sliding":
                        n_reads += run_chunk_sliding(
                            model, mapping_dict_to_class, prepared["chunk_df"], prepared["seq_len"], tokenizer,
                            args.batch_size, args.stride_aa, dtype, outfile_gff, pbar=pbar
                        )
                    else:
                        chunk_reads, step = run_prepared_chunk(
                            model, label_lut, prepared, dtype, outfile_gff, args.batch_size,
                            amp_dtype=amp_dtype, pbar=pbar,
                            start_step=step, mem_log_interval=args.mem_log_interval,
                        )
                        n_reads += chunk_reads

                    # No clear_memory() here: the chunk's NumPy arrays are freed by
                    # refcount as soon as `prepared` is rebound, and the GPU allocator
                    # reuses its (tiny, steady-state) pool across batches.
                    del prepared

            outfile_gff.close()

            elapsed = time.time() - sample_start
            reads_per_s = n_reads / elapsed if elapsed > 0 else 0.0
            print(f"{sample} (DeepCDS {cfg['label']}): {n_reads} reads in {elapsed:.1f}s ({reads_per_s:.1f} reads/s)", flush=True)

            with open(timing_log_path, "a") as timing_log:
                timing_log.write(f"{sample}\t{error_type}\t{n_reads}\t{elapsed:.2f}\t{reads_per_s:.2f}\n")

        del model
        clear_memory(sync=True)
