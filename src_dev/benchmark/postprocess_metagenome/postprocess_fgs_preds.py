"""
Postprocess FragGeneScanRs predictions on the CAMI metagenome test set (raw_predictions
from predict_metagenome/predict_with_fgs.py) into the same {sample}.pkl format
postprocess/postprocess_fgs_preds.py produces for the Mason-simulated genome testsets.

Two differences from the original, both driven by CAMI reads being real reads rather
than fixed-length simulated ones:
  - seq_len (used only for the "off by 3" FGS end-coordinate bug fix) can't be a single
    per-testset constant here, since CAMI reads vary in length. It's looked up per read
    from the sample's <sample>_cds_labels.tsv.gz (produced by process_and_map_CAMI_testset.py)
    instead.
  - The with-errors testset_type/error_model split is instead CAMI_metagenome_{error_model}
    (see predict_with_fgs.py's error_models = ["complete", "illumina_5", "illumina_10"]),
    discovered dynamically rather than assumed, and CAMI samples are discovered the same
    way rather than taken from the fixed genome test_partition_accessions.txt.

get_cds_chunks_on_read/trim_cds are unchanged from the original - purely CIGAR/indel-position
based, no dependency on accession or fixed read length.

EXCLUDE_UNCERTAIN_REGION_READS (below) mirrors postprocess_metagenome_testset.py's
toggle of the same name: when on, any read that postprocess_metagenome_testset.py
would drop from the test set (uncertain_region_overlap=True - overlapping a
hypothetical-protein / ab-initio-prediction-tagged CDS) is also dropped here from the
predictions, so a model isn't penalized as a false positive for calling a CDS on a
read whose ground truth we've decided not to trust.

EXCLUDE_NO_CDS_CONTIG_READS (below) mirrors postprocess_metagenome_testset.py's toggle
of the same name: when on, predictions on reads whose contig has zero CDS annotations
at all (see load_no_cds_contigs / test_CAMI/cami_contigs_without_cds.py) are dropped too.
Independent of and composable with EXCLUDE_UNCERTAIN_REGION_READS.
"""
import argparse
import glob
import os
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# SCARB cluster data root - matches postprocess/postprocess_fgs_preds.py and every
# other benchmark/postprocess script's project_root convention.
project_root = "/tmp/nrt204/FragmentPredictor"

data_dir = "CAMI_metagenome"  # matches predict_with_prodigal.py / predict_with_fgs.py / predict_with_DeepCDS.py

# See postprocess_metagenome_testset.py's toggles of the same names.
EXCLUDE_UNCERTAIN_REGION_READS = False
EXCLUDE_NO_CDS_CONTIG_READS = False


def load_no_cds_contigs(sample, project_root=project_root):
    """Contig accessions with zero CDS features in their raw GFF3 at all, persisted by
    test_CAMI/cami_contigs_without_cds.py."""
    contigs_path = Path(project_root) / "data" / "processed_data" / "CAMI_metagenome_no_cds_contigs" / f"{sample}_contigs_without_cds.txt"
    if not contigs_path.exists():
        raise FileNotFoundError(
            f"{contigs_path} not found - run test_CAMI/cami_contigs_without_cds.py --sample {sample} first."
        )
    return set(contigs_path.read_text().split())


def trim_cds(start, stop, frame):
    """
    Trim CDS coordinates [start, stop] (1-indexed) to include
    only complete codons in the correct reading frame.

    Args:
        start (int): Start coordinate (1-indexed)
        stop (int): Stop coordinate (end position in last coding codon; 1-indexed)
        frame (int): Reading frame (0, 1, or 2)

    Returns:
        Trimmed inputs (start, stop, frame)
    """

    # Adjust start forward to first valid codon in frame
    while (start - frame) % 3 != 1:
        start += 1

    # Adjust stop backward to preserve full codons
    while (stop - start + 1) % 3 != 0:
        stop -= 1

    return start, stop, frame


def get_cds_chunks_on_read(cds_start_read, cds_end_read, insertions, deletions, initial_rf):
    """
    Map back the chunks of the CDS to the original read.
    When an indel has been predicted, FGS returns the full CDS with the initital reading frame and with CDS coordinates on the read it was derived from.
    instead of the fragmented CDS coordinates within different reading frames, which is why this postprocessing is needed.
    Indel positions are excluded from the fragmented CDS chunks.

    Args:
        cds_start_read (int): The CDS start position on the read.
        cds_end_read (int): The CDS stop position on the read.
        insertions: positions that FGS predicts a nucleotide was inserted on the read
        deletions: positions that FGS predicts a nucleotide was deleted on the read
        initial_rf (0, 1, 2): The reading frame that the CDS begins in before any indel errors.

    Returns:
        corrected_complete_cds_coords (list of lists): The corrected, complete CDS fragments belonging to the same overall CDS

    """

    # Combine all indel positions and sort them (the points where we must break the sequence)
    break_points = sorted(set(insertions + deletions))

    # Initialize
    chunks = []
    current_start = cds_start_read
    current_rf = initial_rf
    corrected_complete_cds_coords = []

    # Process all break points plus the final end point
    all_breaks = break_points + [cds_end_read + 1]  # +1 to include the end in the last loop

    # Iterate through each break point
    for break_pos in all_breaks:
        # The end of the current contiguous chunk is the position BEFORE the break
        chunk_end = break_pos - 1

        # Only create a chunk if the current start is before or equal to the calculated end
        if current_start <= chunk_end and current_start <= cds_end_read and chunk_end >= cds_start_read:
            # Cut the chunk to the CDS boundaries
            effective_start = max(current_start, cds_start_read)
            effective_end = min(chunk_end, cds_end_read)
            if effective_start <= effective_end:
                chunks.append([effective_start, effective_end, current_rf])

        # update the reading frame for the next CDS chunk based on what type of break this is
        if break_pos in deletions:
            # A deletion means we are MISSING a base at break_pos in the read -> frame pushed forward by 1
            current_rf = (current_rf - 1) % 3
            # The next chunk starts at the break_pos itself;
            # The base that should be at break_pos is missing, so the next available base is at break_pos.
            current_start = break_pos

        elif break_pos in insertions:
            # An insertion means we have an EXTRA base at break_pos in the read;
            # We skip it, so the next base in the read is at break_pos + 1 -> frame psuhed back by 1
            current_rf = (current_rf + 1) % 3
            # The next chunk starts after the inserted base.
            current_start = break_pos + 1

        # end of CDS
        else:
            current_start = break_pos

    # Trim fragmented CDS coords to include only complete codons
    for cds_coords_fragment in chunks:
        cds_start = cds_coords_fragment[0]
        cds_stop = cds_coords_fragment[1]
        rf = cds_coords_fragment[2]

        cds_start, cds_stop, rf = trim_cds(cds_start, cds_stop, rf)

        # Add CDS fragments except fragmented predicted codons caused by predicted indels placed in close proximity
        if (abs(cds_stop - cds_start) + 1) % 3 == 0:
            corrected_complete_cds_coords.append([cds_start, cds_stop, str(rf)])
        else:
            assert cds_stop - cds_start <= 1  # Ensure that only disrupted codons (spanning maximum 2 nucleotide positions) are removed

    return corrected_complete_cds_coords


def remap_and_process_fgs_preds(sample, pred_dir, read_lengths, excluded_reads=frozenset()):
    """
    Remap predicted CDS coordinates to match format of test set for benchmark.

    Args:
        sample (str): The CAMI sample identifier, e.g. "sample_0".
        pred_dir (str): Raw-predictions subdirectory, e.g. "CAMI_metagenome_illumina_5".
        read_lengths (dict): read_name -> full read length, for the FGS end-coordinate bug fix.
        excluded_reads: read_names to drop from the result entirely (see
            EXCLUDE_UNCERTAIN_REGION_READS).
    """

    # Load predicted fragments with indel error
    preds_info_path = f"{project_root}/data/processed_data/predictions/raw_predictions/fgs_preds/{pred_dir}/{sample}/{sample}.out"

    # Initialize
    fgs_preds_dict = dict()
    seq_len = None

    # Iterate over each read prediction information in the preds_info file
    with open(preds_info_path, 'r') as preds_info_file:
        for line in preds_info_file:

            # Entry line for read
            if line.startswith('>'):
                # Extract read ID
                read_id = line[1:].strip().split("|")[0]
                fgs_preds_dict[read_id] = dict()
                fgs_preds_dict[read_id]["cds_coords"] = []
                fgs_preds_dict[read_id]["cds_fragments_connection"] = []
                fgs_preds_dict[read_id]["seq_error_positions"] = []
                cds_fragments_connections_counter = 0
                seq_len = read_lengths.get(read_id)

            # Prediction lines for read (each line contains predictions for one CDS; there can be mutiple on one read)
            else:
                cds_coord_info = line.strip().split("\t")
                # All reads have been processed to "template" strand; remove those detected on complement strand
                if cds_coord_info[2] == "+":
                    # Get start and stop coordinate for CDS
                    start_coord = int(cds_coord_info[0])
                    end_coord = int(cds_coord_info[1])

                    # ADDED DUE TO BUG IN FGS
                    if seq_len is not None and end_coord == seq_len - 3:
                        end_coord = seq_len

                    # rf in fgs prediction files are (1, 2, 3); change to (0, 1, 2)
                    rf = int(cds_coord_info[3]) - 1

                    # get all predicted insertions and deletions involved with predicted CDS and turn into lists
                    insertions = cds_coord_info[5].strip("I:").split(",")
                    deletions = cds_coord_info[6].strip("D:").split(",")
                    insertions_list = [int(insertion_pos) for insertion_pos in insertions if insertion_pos != ""]
                    deletions_list = [int(deletion_pos) for deletion_pos in deletions if deletion_pos != ""]

                    # In no insertions and deletions are predicted in CDS, just return predicted coords
                    if insertions_list == [] and deletions_list == []:
                        coord_info = [start_coord, end_coord, str(rf)]
                        fgs_preds_dict[read_id]["cds_coords"].append(coord_info)
                        fgs_preds_dict[read_id]["cds_fragments_connection"].append([cds_fragments_connections_counter])
                        cds_fragments_connections_counter += 1

                    # Run over reads containing indels
                    else:
                        # Get CDS fragments interrupted due to indels
                        mapped_cds_fragments_on_read = get_cds_chunks_on_read(start_coord, end_coord, insertions_list, deletions_list, rf)
                        fgs_preds_dict[read_id]["cds_coords"] += mapped_cds_fragments_on_read

                        # Store information of which disrupted fragments belong "together"
                        cds_frags = len(mapped_cds_fragments_on_read)
                        cds_frags_pos = []

                        for i in range(cds_frags):
                            cds_frags_pos.append(cds_fragments_connections_counter)
                            cds_fragments_connections_counter += 1

                        fgs_preds_dict[read_id]["cds_fragments_connection"].append(cds_frags_pos)

                        # Store predicted indel positions
                        indels = [str(insertion_pos) + "I" for insertion_pos in insertions_list] + [str(deletion_pos) + "D" for deletion_pos in deletions_list]
                        fgs_preds_dict[read_id]["seq_error_positions"] += indels

    # Filter out all read ids with CDS predicted only on complement strand
    fgs_preds_dict = {
        read_id: data
        for read_id, data in fgs_preds_dict.items()
        if data["cds_coords"] != [] and read_id not in excluded_reads}

    return fgs_preds_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-uncertain-region-reads", action="store_true", default=EXCLUDE_UNCERTAIN_REGION_READS,
                         help="Override EXCLUDE_UNCERTAIN_REGION_READS (module default: %(default)s)")
    parser.add_argument("--exclude-no-cds-contig-reads", action="store_true", default=EXCLUDE_NO_CDS_CONTIG_READS,
                         help="Override EXCLUDE_NO_CDS_CONTIG_READS (module default: %(default)s)")
    args = parser.parse_args()
    EXCLUDE_UNCERTAIN_REGION_READS = args.exclude_uncertain_region_reads
    EXCLUDE_NO_CDS_CONTIG_READS = args.exclude_no_cds_contig_reads

    input_dir = f"{project_root}/data/processed_data/reads_processed/test/{data_dir}"
    sample_tsv_paths = sorted(glob.glob(f"{input_dir}/*_cds_labels.tsv.gz"))
    if not sample_tsv_paths:
        raise FileNotFoundError(f"No *_cds_labels.tsv.gz files found in: {input_dir}")

    fgs_root = f"{project_root}/data/processed_data/predictions/raw_predictions/fgs_preds"
    pred_dirs = sorted(
        d for d in os.listdir(fgs_root)
        if d.startswith(f"{data_dir}_") and os.path.isdir(os.path.join(fgs_root, d))
    )
    if not pred_dirs:
        raise FileNotFoundError(f"No {data_dir}_* prediction directories found in: {fgs_root}")
    print(pred_dirs)

    # Output-only directory name - raw predictions are unaffected by the toggles, only
    # which reads make it into processed_predictions/ here. Keeps all versions from
    # overwriting each other so they remain available for comparison.
    out_data_dir = (
        data_dir
        + ("_no_ab_initio" if EXCLUDE_UNCERTAIN_REGION_READS else "")
        + ("_no_cds_contigs" if EXCLUDE_NO_CDS_CONTIG_READS else "")
    )
    print(f"EXCLUDE_UNCERTAIN_REGION_READS={EXCLUDE_UNCERTAIN_REGION_READS}, "
          f"EXCLUDE_NO_CDS_CONTIG_READS={EXCLUDE_NO_CDS_CONTIG_READS} -> writing to processed_predictions/fgs_preds/{out_data_dir}_*/")

    for sample_tsv_path in tqdm(sample_tsv_paths, desc="Processing CAMI samples..."):
        sample = os.path.basename(sample_tsv_path).replace("_cds_labels.tsv.gz", "")
        print(sample)

        tsv_columns = ["read_name", "read"] + (["uncertain_region_overlap"] if EXCLUDE_UNCERTAIN_REGION_READS else []) \
            + (["contig_accession"] if EXCLUDE_NO_CDS_CONTIG_READS else [])
        tsv_df = pd.read_csv(sample_tsv_path, sep="\t", compression="gzip", usecols=tsv_columns)
        read_lengths = tsv_df.set_index("read_name")["read"].str.len().to_dict()

        excluded_reads = set()
        if EXCLUDE_UNCERTAIN_REGION_READS:
            excluded_reads |= set(tsv_df.loc[tsv_df["uncertain_region_overlap"], "read_name"])
        if EXCLUDE_NO_CDS_CONTIG_READS:
            no_cds_contigs = load_no_cds_contigs(sample)
            excluded_reads |= set(tsv_df.loc[tsv_df["contig_accession"].isin(no_cds_contigs), "read_name"])

        for pred_dir in pred_dirs:
            out_pred_dir = out_data_dir + pred_dir[len(data_dir):]  # e.g. CAMI_metagenome_illumina_5 -> {out_data_dir}_illumina_5
            out_dir = f"{project_root}/data/processed_data/predictions/processed_predictions/fgs_preds/{out_pred_dir}"
            os.makedirs(out_dir, exist_ok=True)

            fgs_preds_dict = remap_and_process_fgs_preds(sample, pred_dir, read_lengths, excluded_reads=excluded_reads)

            with open(f"{out_dir}/{sample}.pkl", "wb") as processed_preds_file:
                pickle.dump(fgs_preds_dict, processed_preds_file)
