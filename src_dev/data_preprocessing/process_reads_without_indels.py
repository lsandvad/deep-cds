import gzip
import os
import re
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import pysam
from Bio import BiopythonDeprecationWarning, SeqIO

# Suppress FASTA comment deprecation warnings globally
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

compute_machine = "cluster"  # Options: "cluster" or "local"

if compute_machine == "cluster":
    data_base_path = "/tmp/nrt204/FragmentPredictor" #ERDA mount point on cluster   
else:
    data_base_path = "../.."

accessions_train = open(f"{data_base_path}/data/processed_data/genome_partitions/train_partition_accessions.txt").read().splitlines()
accessions_val = open(f"{data_base_path}/data/processed_data/genome_partitions/val_partition_accessions.txt").read().splitlines()
accessions_test = open(f"{data_base_path}/data/processed_data/genome_partitions/test_partition_accessions.txt").read().splitlines()


###Functions for processing CDS annotations###
def get_assembly_details(accession, assembly_details):
    """
    Get assembly names of all genomic sequences in fasta file.

    Args:
        accession (str): The accession name of a species
        assembly_details (dict): a dictionary, either empty or with information (iterated over for both refseq and genbank).

    Returns:
        assembly_details (dict): a dictionary with length of genomic sequence as key and assembly name for that sequence as value.
    """

    # Read the genome file
    genome_files = os.listdir(f"{data_base_path}/data/raw_data/genome_data/{accession}/")
    genome_fasta_file = [file for file in genome_files if file.startswith(accession)][0]
    genome_filename = f"{data_base_path}/data/raw_data/genome_data/{accession}/{genome_fasta_file}"

    try:
        # Loop over sequences in the genome file
        for record in SeqIO.parse(genome_filename, "fasta"):
            assembly_len = len(record.seq)
            assembly_id = record.id
            if assembly_len not in assembly_details.keys():
                assembly_details[assembly_len] = []
            assembly_details[assembly_len].append(assembly_id)

    except ValueError or KeyError as err:
        # Loop over sequences in the genome file
        for record in SeqIO.parse(genome_filename, "fasta-blast"):
            assembly_len = len(record.seq)
            assembly_id = record.id
            if assembly_len not in assembly_details.keys():
                assembly_details[assembly_len] = []
            assembly_details[assembly_len].append(assembly_id)

    return assembly_details


def quality_check_cds_annotations(cds_coords, cds_coords_uncertain):
    """
    Perform a quality check on extracted CDS annotations.
    If any annotations do not pass a quality check, they are moved to the group of uncertain annotations.

    Args:
        cds_coords (dict): dictionary containing start and stop coordinates of annotated CDS with assembly names as keys.
        cds_coords_uncertain (dict): dictionary containing uncertain start and stop coordinates of annotated CDS with assembly names as keys.

    Returns:
        cds_coords (dict): Updated version.
        cds_coords_uncertain (dict): Updated version.
    """

    # Remove duplicate annotations
    for assembly in cds_coords.keys():
        cds_coords[assembly] = set(tuple(coord_set) for coord_set in cds_coords[assembly])  # Remove duplicate coordinate sets
        cds_coords[assembly] = sorted([list(coord_set) for coord_set in cds_coords[assembly]])  # Turn back into sorted list

        # Find coordinate sets with same start position but different end positions;
        # This happens when a stop codon translates into selenocystein. Extract longest CDS
        start_to_ends = defaultdict(list)
        for start, end in cds_coords[assembly]:
            start_to_ends[start].append(end)

        # Find conflicting annotations: same start, different end
        conflicts_end = {start: ends for start, ends in start_to_ends.items() if len(ends) > 1}

        # Clean up conflicting annotations
        for start, potential_ends in conflicts_end.items():
            end = max(potential_ends)  # Extract stop coordinate giving the longest CDS
            truncated_end = min(potential_ends)

            # Check the reading frame length is a multiple of 3 (end -2 due to coordinate end in last position of stop codon)
            if ((end - 2) - start) % 3 != 0:
                cds_coords[assembly].remove([start, end])
                cds_coords_uncertain[assembly].append([start, end])

            # Remove truncated version of CDS
            cds_coords[assembly].remove([start, truncated_end])

        # Find coordinate sets with same end position, different start positions: keep coordinate set with longest CDS
        end_to_starts = defaultdict(list)
        for start, end in cds_coords[assembly]:
            end_to_starts[end].append(start)

        # Find conflicting annotations: different start, same end
        conflicts_start = {end: starts for end, starts in end_to_starts.items() if len(starts) > 1}

        # Clean up conflicting annotations
        for end, potential_starts in conflicts_start.items():
            start = min(potential_starts)  # extract start coordinate giving the longest CDS
            truncated_start = max(potential_starts)

            # Check the reading frame length is a multiple of 3 (end -2 due to coordinate end in last position of stop codon)
            if ((end - 2) - start) % 3 != 0:
                cds_coords[assembly].remove([start, end])
                cds_coords_uncertain[assembly].append([start, end])

            # Remove truncated version of CDS
            cds_coords[assembly].remove([truncated_start, end])

        # Check that remaining CDS annotations gives a reading frame length that is a multiple of 3
        # (end -2 due to coordinate end in last position of stop codon)
        for cds in cds_coords[assembly]:
            if ((cds[1] - 2) - cds[0]) % 3 != 0:
                cds_coords[assembly].remove(cds)
                cds_coords_uncertain[assembly].append(cds)

    return cds_coords, cds_coords_uncertain


def convert_complement_coordinates(refseq_accession, cds_coords_complement, cds_coords_uncertain_complement):
    """
    Convert CDS annotations present on complement strand to fit position of reverse-complemented genomic sequence.

    Args:
        refseq_accession: The RefSeq accession name
        cds_coords_complement (dict): dictionary containing start and stop coordinates of annotated CDS on
                                      compelement strand with assembly names as keys.
        cds_coords_uncertain_complement (dict): dictionary containing uncertain start and stop coordinates
                                                of annotated CDS on complement strand with assembly names as keys.

    Returns:
        cds_coords_complement (dict): As input, but with coordinate sets converted.
        cds_coords_uncertain_complement (dict): As input, but with coordinate sets converted.
    """

    # Extract filename and -path of file with genomic sequence(s) (assemblies: chromosome(s), plasmid(s))
    genome_files = os.listdir(f"{data_base_path}/data/raw_data/genome_data/{refseq_accession}/")
    genome_fasta_file = [file for file in genome_files if file.startswith(refseq_accession)][0]
    genome_filename = f"{data_base_path}/data/raw_data/genome_data/{refseq_accession}/{genome_fasta_file}"

    # Loop over each assembly sequence in genomic file
    for record in SeqIO.parse(genome_filename, "fasta"):
        assembly_id = record.id
        seq_len = len(record.seq)

        try:
            coord_sets_assembly = sorted(cds_coords_complement[assembly_id], reverse=True)
            coord_sets_uncertain_assembly = sorted(cds_coords_uncertain_complement[assembly_id], reverse=True)
        except KeyError:
            print("No CDS annotations on assembly.")

        # Reset cds coordinates for complement strand
        cds_coords_complement[assembly_id] = []
        for cds in coord_sets_assembly:
            cds_coords_complement[assembly_id].append([seq_len - cds[1] + 1, seq_len - cds[0] + 1])

        # Reset cds coordinates for complement strand
        cds_coords_uncertain_complement[assembly_id] = []
        for cds in coord_sets_uncertain_assembly:
            cds_coords_uncertain_complement[assembly_id].append([seq_len - cds[1] + 1, seq_len - cds[0] + 1])

    return cds_coords_complement, cds_coords_uncertain_complement


def extract_CDS(refseq_accession):
    """
    Extracts the CDS coordinates from RefSeq and GenBank GFF files (annotation data).
    The GenBank accession number is found implicitly from the RefSeq accession number.

    Args:
        refseq_accession (str): The RefSeq accession number.

    Returns:
        cds_coords_genbank_template (list): CDS coordinates on template strand from the GenBank GFF file.
        cds_coords_refseq_template (list): CDS coordinates on template strand from the RefSeq GFF file.
    """

    # Get GenBank accession from RefSeq accession
    genbank_accession = refseq_accession.replace("GCF_", "GCA_")

    # Connect assembly names from RefSeq and GenBank annotations of matching genomic sequencing
    assembly_details = dict()
    assembly_details = get_assembly_details(refseq_accession, assembly_details)
    try:
        assembly_details = get_assembly_details(genbank_accession, assembly_details)
    except FileNotFoundError:
        print("No GenBank assembly details available.")

    # Set assembly name of GenBank as key, corresponding assembly name of RefSeq as value.
    assembly_conversion = {values[-1]: values[0] for values in assembly_details.values()}

    # Initialize
    cds_coords_template = dict()
    cds_coords_uncertain_template = dict()
    cds_coords_complement = dict()
    cds_coords_uncertain_complement = dict()

    # When available, extract GenBank annotation data
    try:
        with open(f"{data_base_path}/data/raw_data/genome_data/{genbank_accession}/genomic.gff", "r") as gff_file_genbank:
            for line in range(7):
                next(gff_file_genbank)
            for line in gff_file_genbank:
                # Find all GenBank annotated CDS on template strand
                if "\tCDS\t" in line and "\t+\t" in line:
                    assembly_id = line.split("\t")[0]

                    # Move uncertain, incomplete or predicted CDS annotations for assembly into dict of uncertain annotations
                    if (
                        "pseudo=true" in line.lower()
                        or "product=hypothetical protein" in line.lower()
                        or "partial=true" in line.lower()
                        or "ab initio prediction" in line.lower()
                        or "note=programmed frameshift" in line.lower()
                    ):
                        if assembly_conversion[assembly_id] not in cds_coords_uncertain_template.keys():
                            cds_coords_uncertain_template[assembly_conversion[assembly_id]] = []
                        cds_coords_uncertain_template[assembly_conversion[assembly_id]].append([int(line.split("\t")[3]), int(line.split("\t")[4])])

                    # Store certain coordinate sets for assembly in dict
                    else:
                        if "pseudogene" not in line.lower():
                            if assembly_conversion[assembly_id] not in cds_coords_template.keys():
                                cds_coords_template[assembly_conversion[assembly_id]] = []
                            cds_coords_template[assembly_conversion[assembly_id]].append([int(line.split("\t")[3]), int(line.split("\t")[4])])

                # Same procedure on complement strand
                elif "\tCDS\t" in line and "\t-\t" in line:
                    assembly_id = line.split("\t")[0]
                    if (
                        "pseudo=true" in line.lower()
                        or "product=hypothetical protein" in line.lower()
                        or "partial=true" in line.lower()
                        or "ab initio prediction" in line.lower()
                        or "note=programmed frameshift" in line.lower()
                    ):
                        if assembly_conversion[assembly_id] not in cds_coords_uncertain_complement.keys():
                            cds_coords_uncertain_complement[assembly_conversion[assembly_id]] = []
                        cds_coords_uncertain_complement[assembly_conversion[assembly_id]].append([int(line.split("\t")[3]), int(line.split("\t")[4])])

                    else:
                        if "pseudogene" not in line.lower():
                            if assembly_conversion[assembly_id] not in cds_coords_complement.keys():
                                cds_coords_complement[assembly_conversion[assembly_id]] = []
                            cds_coords_complement[assembly_conversion[assembly_id]].append([int(line.split("\t")[3]), int(line.split("\t")[4])])

    except FileNotFoundError:
        print("No GenBank annotation data available.")

    # When available, extract RefSeq annotation data
    try:
        with open(f"{data_base_path}/data/raw_data/genome_data/{refseq_accession}/genomic.gff", "r") as gff_file_refseq:
            for line in range(9):
                next(gff_file_refseq)
            for line in gff_file_refseq:
                # Find all RefSeq annotated CDS on template strand
                if "\tCDS\t" in line and "\t+\t" in line:
                    assembly_id = line.split("\t")[0]

                    # Move uncertain, incomplete or predicted CDS annotations for assembly into dict of uncertain annotations
                    if (
                        "pseudo=true" in line.lower()
                        or "product=hypothetical protein" in line.lower()
                        or "partial=true" in line.lower()
                        or "ab initio prediction" in line.lower()
                        or "note=programmed frameshift" in line.lower()
                    ):
                        if assembly_id not in cds_coords_uncertain_template.keys():
                            cds_coords_uncertain_template[assembly_id] = []
                        cds_coords_uncertain_template[assembly_id].append([int(line.split("\t")[3]), int(line.split("\t")[4])])

                    # Store certain coordinate sets for assembly in dict
                    else:
                        if "pseudogene" not in line.lower():
                            if assembly_id not in cds_coords_template.keys():
                                cds_coords_template[assembly_id] = []
                            cds_coords_template[assembly_id].append([int(line.split("\t")[3]), int(line.split("\t")[4])])

                # Same procedure for complement strand
                elif "\tCDS\t" in line and "\t-\t" in line:
                    assembly_id = line.split("\t")[0]
                    if (
                        "pseudo=true" in line.lower()
                        or "product=hypothetical protein" in line.lower()
                        or "partial=true" in line.lower()
                        or "ab initio prediction" in line.lower()
                        or "note=programmed frameshift" in line.lower()
                    ):
                        if assembly_id not in cds_coords_uncertain_complement.keys():
                            cds_coords_uncertain_complement[assembly_id] = []
                        cds_coords_uncertain_complement[assembly_id].append([int(line.split("\t")[3]), int(line.split("\t")[4])])

                    else:
                        if "pseudogene" not in line.lower():
                            if assembly_id not in cds_coords_complement.keys():
                                cds_coords_complement[assembly_id] = []
                            cds_coords_complement[assembly_id].append([int(line.split("\t")[3]), int(line.split("\t")[4])])

    except FileNotFoundError:
        print("No RefSeq annotation data available.")

    # Convert complement-strand CDS annotations to match position on reverse-complemented genomic sequences
    cds_coords_complement, cds_coords_uncertain_complement = convert_complement_coordinates(refseq_accession, cds_coords_complement, cds_coords_uncertain_complement)

    # Quality check annotated CDSs
    cds_coords_template, cds_coords_uncertain_template = quality_check_cds_annotations(cds_coords_template, cds_coords_uncertain_template)
    cds_coords_complement, cds_coords_uncertain_complement = quality_check_cds_annotations(cds_coords_complement, cds_coords_uncertain_complement)

    return cds_coords_template, cds_coords_uncertain_template, cds_coords_complement, cds_coords_uncertain_complement


###Process reads and store necessary information###
def read_is_in_uncertain_range(coordinate, uncertain_coordinates_list, seqs_len):
    """
    Identify whether the given read is in range of an 'uncertain' region (defined by the quality checks of annotated CDSs).

    Args:
        coordinate (int): the start coordinate of the given read
        coordinates_list (list): list of uncertain coordinates
        seqs_len (int): length of reads in dataset

    Returns:
        True if read in within 'uncertain' range
    """
    for uncertain_coords in uncertain_coordinates_list:
        if uncertain_coords[0] - seqs_len <= coordinate <= uncertain_coords[1]:
            return True


def proces_reads_to_dict(accession, seqs_len, cds_coords_uncertain, strand, partition, error_rates):
    """
    Save reads with necessary information from .bam-file (Mason output) to dict, discard reads of 'uncertain' areas.

    Args:
        accession (str): The RefSeq accession name.
        seqs_len (int): The length of simulated reads.
        cds_coords_uncertain (dict): dict with coordinates of 'uncertain' CDS annotations.
        strand (str): Whether to process reads generated from template or complement strand.
                      Options: "template_strand" or "complement strand".

    Returns:
        reads_information_dict (dict): nested dictionary with necessary information on each read, divided on each assembly ID
    """

    # Initialize
    reads_information_dict = {}

    # Open Mason output information (.bam-file)
    with pysam.AlignmentFile(f"{data_base_path}/data/processed_data/simulated_reads/{partition}/{error_rates}/{strand}/alignments/{accession}_alignments.bam", "rb") as bam_infile:
        for read in bam_infile:
            # Extract information from the read
            start_coordinate = read.reference_start + 1  # pysam is 0-based, convert to 1-base
            assembly = read.reference_name

            # Ignore reads that span uncertain CDS annotation areas
            if assembly in cds_coords_uncertain:
                if read_is_in_uncertain_range(start_coordinate, cds_coords_uncertain[assembly], seqs_len):
                    continue  # Do not keep reads overlapping with uncertain annotation areas; continue

            # Initialize nested dict for assembly
            if assembly not in reads_information_dict:
                reads_information_dict[assembly] = {}

            # Get read-specific information
            read_id = read.query_name
            CIGAR = read.cigarstring
            read_seq = read.query_sequence
            md_z = read.get_tag("MD") if read.has_tag("MD") else None  # Handle missing MD tag

            # Write read-specific information to dict
            reads_information_dict[assembly][read_id] = {"CIGAR": CIGAR, "start_coordinate": start_coordinate, "read": read_seq, "MD:Z": md_z}

    return reads_information_dict


def get_position_gene_overlaps(cds_ranges, start_coord, length):
    """
    Find the specific CDS coordinates (if any) that overlaps with positions in a read (read positions defined as start_coord to start_coord+length).

    Args:
        cds_ranges (list): List of [[start, end],...] coordinates for genes
        start_coord (int): Starting coordinate to check (on genomic scale)
        length (int): Length of the region to check

    Returns:
        overlaps (dict): gene IDs as names, list of CDS [start, stop] coordinates overlapping with read as values.
    """

    end_coord = start_coord + length
    overlaps = {}

    # Loop over each CDS range
    for i, (gene_start, gene_end) in enumerate(cds_ranges):
        # Check if there is any overlap
        if not (end_coord < gene_start or start_coord > gene_end):
            # Store overlapping genes for subsequence region (read)
            gene_key = f"g{i}"
            overlaps[gene_key] = [gene_start, gene_end]

    return overlaps


def generate_rf_labels(sequence_start, sequence_length, cds_coords):
    """
    Generate reading frame labels for a sequence, accounting for multiple potential CDSs.

    Args:
        sequence_start (int): Start coordinate of the sequence.
        sequence_length (int): Length of the sequence (e.g., 300).
        cds_coords (dict): Dictionary of CDSs in format {'id': [start, end]}.

    Returns:
        rf_labels_aa_level (dict): dict with read positions as keys, labels as values for each amino acid in reading frame
        rf_labels_nt_level (dict): dict with read positions as keys, labels as values for each nucleotide in reading frame
              Labels: 0 (non-coding), 1 (coding), 2 (start), 3 (stop).
    """

    sequence_end = sequence_start + sequence_length - 1

    # 0-index
    sequence_start = sequence_start - 1
    sequence_end = sequence_end - 1

    # Initialize list for reading frame labels
    rf_labels = [0] * sequence_length

    # Loop over each CDS ID and corresponding start and stop coordinates
    for cds_id, (cds_start, cds_end) in cds_coords.items():
        # 0-index gene positions
        cds_start -= 1
        cds_end -= 1

        # Skip CDSs entirely not overlapping with read positions
        if cds_end < sequence_start or cds_start > sequence_end:
            continue

        # Calculate overlap positions between read and CDSs
        overlap_start = max(cds_start, sequence_start)
        overlap_end = min(cds_end, sequence_end)

        # Determine reading frame (RF) of the CDS (0, 1, or 2)
        # Later in code, cds_start is adapted to each RF (+1 for RF1, +2 for RF2),
        # meaning that we will identify the correct RF as having offset = 0 in either case.
        offset = (cds_start - sequence_start) % 3
        if offset == 0:
            # Mark coding regions (1), start (2), and stop (3)
            for pos in range(overlap_start, overlap_end + 1):
                seq_pos = pos - sequence_start
                if seq_pos < 0 or seq_pos >= sequence_length:
                    continue  # Sanity check

                # Start codon (first 3 bases of the CDS)
                if pos in range(cds_start, cds_start + 3):
                    rf_labels[seq_pos] = 2

                # Stop codon (last 3 bases of the CDS)
                elif pos in range(cds_end - 2, cds_end + 1):
                    rf_labels[seq_pos] = 3

                # Coding region (not start/stop)
                else:
                    if rf_labels[seq_pos] == 0:  # Don't overwrite start/stop
                        rf_labels[seq_pos] = 1

    # Nucleotide-level labels will always come with every three labels being the same (corresponding to a codon);
    # extract the first label for each codon to get amino acid-level reading frame labels
    rf_labels_aa_level = rf_labels[::3]
    rf_labels_nt_level = rf_labels

    return rf_labels_aa_level, rf_labels_nt_level


def encode_nucleotide_to_amino_acid(sequence):
    """
    The function takes a nucleotide sequence and translates it into the corresponding amino acid sequence.

    Args:
        sequence (str): A nucleotide sequence.

    Returns:
        amino_acid_sequence (str): A string representing the amino acid sequence translated from the
                                   nucleotide input. Each codon is mapped to its corresponding amino acid,
                                   with stop codons represented by "X".
    """

    # Define the genetic code as a dict
    genetic_code = {
        "TTT": "F",
        "TTC": "F",
        "TTA": "L",
        "TTG": "L",
        "TCT": "S",
        "TCC": "S",
        "TCA": "S",
        "TCG": "S",
        "TAT": "Y",
        "TAC": "Y",
        "TAA": "X",
        "TAG": "X",
        "TGT": "C",
        "TGC": "C",
        "TGA": "X",
        "TGG": "W",
        "CTT": "L",
        "CTC": "L",
        "CTA": "L",
        "CTG": "L",
        "CCT": "P",
        "CCC": "P",
        "CCA": "P",
        "CCG": "P",
        "CAT": "H",
        "CAC": "H",
        "CAA": "Q",
        "CAG": "Q",
        "CGT": "R",
        "CGC": "R",
        "CGA": "R",
        "CGG": "R",
        "ATT": "I",
        "ATC": "I",
        "ATA": "I",
        "ATG": "M",
        "ACT": "T",
        "ACC": "T",
        "ACA": "T",
        "ACG": "T",
        "AAT": "N",
        "AAC": "N",
        "AAA": "K",
        "AAG": "K",
        "AGT": "S",
        "AGC": "S",
        "AGA": "R",
        "AGG": "R",
        "GTT": "V",
        "GTC": "V",
        "GTA": "V",
        "GTG": "V",
        "GCT": "A",
        "GCC": "A",
        "GCA": "A",
        "GCG": "A",
        "GAT": "D",
        "GAC": "D",
        "GAA": "E",
        "GAG": "E",
        "GGT": "G",
        "GGC": "G",
        "GGA": "G",
        "GGG": "G",
    }

    if len(sequence) % 3 == 1:
        sequence = sequence[:-1]
    elif len(sequence) % 3 == 2:
        sequence = sequence[:-2]

    # Ensure the sequence length is a multiple of 3
    assert len(sequence) % 3 == 0, "Input sequence length must be a multiple of 3."

    # Initialize
    amino_acid_sequence = ""

    # Iterate through the nucleotide sequence to encode codons
    for i in range(0, len(sequence), 3):
        codon = sequence[i : i + 3]
        # Get amino acid (stop codons represented as X)
        amino_acid = genetic_code.get(codon, "Å")
        amino_acid_sequence += amino_acid

    assert "Å" not in amino_acid_sequence, "Unknown nucleotides present in sequence."

    return amino_acid_sequence


def mark_errors(md_z, seqs_len):
    """
    Mark positions of sequencing errors in a read.

    Args:
        md_z (str): the MD:Z tag. Provides information on positions of substitution errors.
        seqs_len (int): the read length

    Returns:
        errors_str: information string detailing each sequencing error position, each separated by a comma.
    """

    # Initialize
    errors_str = ""
    pos = 0
    buffer = ""
    write_read = True

    # No errors
    if md_z == str(seqs_len):
        return errors_str, write_read

    # Loop through every position as described ni the MD:Z-tag
    for char in md_z:
        if char.isdigit():
            buffer += char  # Build the base count

        # Letters mark sequencing error
        elif char in {"A", "T", "G", "C", "N"}:
            # Process the accumulated base count
            num_bases = int(buffer) if buffer else 0
            buffer = ""  # Reset for next number
            pos += num_bases

            # Mark the error if within bounds
            if pos <= seqs_len:
                errors_str += str(pos + 1) + char + ","  # Mark original nucletoide for substitution

            pos += 1  # Skip the error base

        elif char == "^":
            write_read = False

    return errors_str[:-1], write_read  # remove last comma


def apply_mutations(sequence, errors_str):
    """
    Apply substitutions to a DNA sequence (read) based on input mutation data.

    Args:
        sequence (str): Original DNA sequence (read)
        errors_str (str): Error string detailing position and nucleotide to mutate to (in format "101G,159A,200I,...")

    Returns:
        Modified sequence with mutations applied
    """

    # No sequencing errors in read
    if errors_str == "":
        return sequence

    # Convert sequence to list for easy modification
    seq_list = list(sequence)

    # Process each mutation in read
    for mutation in errors_str.split(","):
        # Extract position and new base
        pos = int(mutation[:-1]) - 1  # Convert to 0-based index
        new_base = mutation[-1]

        # Only apply if position is valid and base is A, C, G, or T
        if 0 <= pos < len(seq_list) and new_base in {"A", "C", "G", "T"}:
            seq_list[pos] = new_base

    # Convert back to string
    return "".join(seq_list)


def extract_coding_sequences(aa_seq, labels):
    """
    Extract coding part of a reading frame marked by label "1" (excl. start- and stop codons).

    Args:
        aa_seq (str): amino acid sequence for reading frame
        labels (list): labels for each amino acid in sequence

    Returns:
        sequences: coding amino acid sequence(s) from reading frame.
    """

    # Initialize
    sequences = []
    current_seq_start = None

    for i, label in enumerate(labels):
        if label == 1:
            # Start a new sequence if no active sequence
            if current_seq_start is None:
                current_seq_start = i
        else:
            # End the current sequence (if exists)
            if current_seq_start is not None:
                sequences.append(aa_seq[current_seq_start:i])
                current_seq_start = None

    # Add the last sequence if it ends at the last label
    if current_seq_start is not None:
        sequences.append(aa_seq[current_seq_start : len(labels)])

    return sequences


def assign_codon_labels(labels):
    """
    Assign labels to each codon in reads with indel errors (shifts reading frame).

    Args:
        labels (list): list of labels for each position in read

    returns:
        codon_labels (list): list of labels for each codon
    """

    # Initialize
    codon_labels = []

    # Loop over each codon
    for i in range(0, len(labels), 3):
        codon = labels[i : i + 3]
        # If 0 (non-coding label) is in codon, always assign codon non-coding status
        if 0 in codon:
            codon_labels.append(0)
        # If not 0 is not codon, append the label of the first codon position
        else:
            codon_labels.append(codon[0])

    return codon_labels


def parse_cigar(cigar):
    """
    Parse CIGAR string into a list of (operation, length) tuples.

    Args:
        cigar (str): CIGAR string

    Returns:
        operations (list): list of tuples showing the order of operations (Match, Insertion, Deletion) and lengths (in positions) of such.
    """

    # Initialize
    operations = []

    for match in re.finditer(r"(\d+)([MID])", cigar):
        length = int(match.group(1))
        operation = match.group(2)
        operations.append((operation, length))

    return operations


def pad_label_positions(labels_rf_nt, rf):
    """
    Pads nucleotide position labels for different reading frames (rf1/rf2) to maintain
    consistent phase at sequence boundaries.

    Args:
        labels_rf_nt (list): List of labels for each nucleotide position
        rf (str): reading frame, either "rf1" or "rf2".

    Returns:
        labels_rf_nt (list): updated labels list.
    """

    if rf == "rf1":
        # Pad start (1 element) based on first position's phase
        if labels_rf_nt[0] in [1, 3]:
            labels_rf_nt = [1] + labels_rf_nt
        elif labels_rf_nt[0] in [0, 2]:
            labels_rf_nt = [0] + labels_rf_nt

        # Pad end (2 elements) based on last position's phase
        if labels_rf_nt[-1] in [0, 3]:
            labels_rf_nt = labels_rf_nt + [0, 0]
        elif labels_rf_nt[-1] in [1, 2]:
            labels_rf_nt = labels_rf_nt + [1, 1]

    elif rf == "rf2":
        # Pad start (2 elements) based on first position's phase
        if labels_rf_nt[0] in [1, 3]:
            labels_rf_nt = [1, 1] + labels_rf_nt
        elif labels_rf_nt[0] in [0, 2]:
            labels_rf_nt = [0, 0] + labels_rf_nt

        # Pad end (1 element) based on last position's phase
        if labels_rf_nt[-1] in [0, 3]:
            labels_rf_nt = labels_rf_nt + [0]
        elif labels_rf_nt[-1] in [1, 2]:
            labels_rf_nt = labels_rf_nt + [1]

    return labels_rf_nt


def mark_intervals(labels_rf0_nt, labels_rf1_nt, labels_rf2_nt, labels_rf0, labels_rf1, labels_rf2):
    """
    Mark CDS interval coordinates on a given read.

    Args:
    labels_rf0_nt (list): Labels for each nucleotide wrt. reading frame 0
    labels_rf1_nt (list): Labels for each nucleotide wrt. reading frame 1
    labels_rf2_nt (list): Labels for each nucleotide wrt. reading frame 2

    Returns:
        intervals (list): Intervals of CDS regions on read, along with the reading frame of the corresponding CDS.
    """

    # Initialize
    intervals = []

    # Loop over each RF
    for rf in ["0", "1", "2"]:
        # Extract labels for the appropriate reading frame
        if rf == "0":
            labels_rf = labels_rf0
        elif rf == "1":
            labels_rf = labels_rf1
        else:
            labels_rf = labels_rf2

        # If all labels are 0, there is no CDS in the paritcular RF of read
        cds_start = None

        if sum(labels_rf) == 0:  # Skip if all zeros
            continue

        # If some CDS region in the RF is found
        for i, num in enumerate(labels_rf):
            if num in {1, 2}:  # Start codons and coding regions
                if cds_start is None:  # Start of a new interval (convert to 1-based)
                    cds_start = i * 3 + int(rf) + 1

            elif num in {3}:  # Stop codons
                cds_end = (i + 1) * 3 + int(rf)

                if cds_start is None:
                    cds_start = cds_end - 2

                assert (cds_end - cds_start + 1) % 3 == 0, "CDS interval length is not a multiple of 3, which is unexpected for CDS regions."
                intervals.append([cds_start, cds_end, rf])  # i is already end+1 in 0-based

                # Re-initialize
                cds_start = None

            else:
                # CDS end due to indel
                if cds_start is not None:  # End of an interval (convert to 1-based)
                    cds_end = i * 3 + int(rf)
                    assert (cds_end - cds_start + 1) % 3 == 0, "CDS interval length is not a multiple of 3, which is unexpected for CDS regions."
                    intervals.append([cds_start, cds_end, rf])  # i is already end+1 in 0-based

                    # Re-initialize
                    cds_start = None

        # Handle case where interval continues to end of list (convert to 1-based)
        if cds_start is not None:
            if rf == "0":
                cds_end = len(labels_rf0_nt)
                intervals.append([cds_start, cds_end, rf])  # len() gives last pos in 1-based
            elif rf == "1":
                cds_end = len(labels_rf1_nt) + 1
                intervals.append([cds_start, cds_end, rf])  # len() gives last pos in 1-based
            elif rf == "2":
                cds_end = len(labels_rf2_nt) + 2
                intervals.append([cds_start, cds_end, rf])
            assert (cds_end - cds_start + 1) % 3 == 0, "CDS interval length is not a multiple of 3, which is unexpected for CDS regions."

    intervals.sort()  # Sorts in-place

    if sum(labels_rf0) + sum(labels_rf1) + sum(labels_rf2) != 0:
        assert intervals != [], f"No intervals found, but some CDS regions were expected. {labels_rf0, labels_rf1, labels_rf2}"

    return intervals


def quality_check_CDS_fragments(coding_seqs_all_read, accession):
    """
    Check that all amino acid fragments labelled as coding (label = 1) are present in the proteome (checks both the GenBank- and RefSeq annotated proteomes)

    Args:
        coding_seqs_all_read (list): coding sequences in list (amino acid fragments)
        accession (str): The refseq accession ID

    Returns:
        write_read (bool): If read passes all checks, it can be written to output file.
    """
    # If some coding fragments are found in read; check that all of such are present in either RefSeq or GenBank genome
    if coding_seqs_all_read != []:
        genbank_accession = accession.replace("GCF_", "GCA_")
        write_read = True
        for coding_seq in coding_seqs_all_read:
            # Replace X (stop codon from table 11) with W (for table 4 compatibility)
            coding_seq = str(coding_seq).replace("X", "W")

            seq_found = False
            for record in SeqIO.parse(f"{data_base_path}/data/raw_data/genome_data/{accession}/protein.faa", "fasta"):
                prot_seq = record.seq
                if coding_seq in prot_seq:
                    seq_found = True
                    break

            # If sequence is not in RefSeq proteome file, then check GenBank proteome file
            if not seq_found:
                try:
                    for record in SeqIO.parse(f"{data_base_path}/data/raw_data/genome_data/{genbank_accession}/protein.faa", "fasta"):
                        prot_seq = record.seq
                        if coding_seq in prot_seq:
                            seq_found = True
                            break
                except FileNotFoundError:
                    seq_found = False

            if not seq_found:
                write_read = False
                break

    else:  # Write fully intergenic sequences
        write_read = True

    return write_read


def process_strand_reads(assembly_id, assembly_seq, accession, seqs_len, cds_coords_strand, reads_information_dict, processed_reads_df, reads_correct, reads_wrong, strand):
    """
    Process all reads on a given strand of a given assembly to extract information for outputs.

    Args:
        assembly_id (str): The ID of a genomic assembly (chromosome, plasmid etc. from genome)
        assembly_seq (str): The sequence of genomic assembly in question
        accession (str): The accession ID of the genome
        seqs_len (int): The length of the reads
        cds_coords (dict): the CDS coordinates present in the genome
        reads_information_dict (dict): processed reads and additional necessary information of the given strand and assembly from Mason output
        processed_reads_df (df): dataframe storing processed data ready for output to final dataset format
        reads_correct (int): Counter for correctly labelled reads (passes all quality checks)
        reads_wrong (int): Counter for wrongly labelled reads (does not pass all quality checks)
        strand (str): the strand reads have been simulated from. Options are "+" or "-".

    Returns:
        processed_reads_df (df): df updated with samples (that passes quality checks) from assembly and strand
        reads_correct (int): updated counter
        reads_wrong (int): updated counter
    """

    try:
        # Extract all CDSs from assembly (strand-specific in input, see run_pipeline())
        cds_coords_assembly = cds_coords_strand[assembly_id]

        # Extract IDs of all reads
        read_ids = list(reads_information_dict[assembly_id].keys())

        # Prepare read processing based on sequence length/3 (ensure complete number of codons being labeled)
        if seqs_len % 3 == 0:
            seqs_len_rf0 = seqs_len
            seqs_len_rf1 = seqs_len - 3
            seqs_len_rf2 = seqs_len - 3

        elif seqs_len % 3 == 1:
            seqs_len_rf0 = seqs_len - 1
            seqs_len_rf1 = seqs_len - 1
            seqs_len_rf2 = seqs_len - 4

        elif seqs_len % 3 == 2:
            seqs_len_rf0 = seqs_len - 2
            seqs_len_rf1 = seqs_len - 2
            seqs_len_rf2 = seqs_len - 2

        # Iterate through each read
        for read_id in read_ids:
            # Initialize for each read
            write_read = None
            cds_overlaps_read = []
            coding_seqs_all_read = []

            # Extract read information
            read = reads_information_dict[assembly_id][read_id]["read"]
            start_coord = reads_information_dict[assembly_id][read_id]["start_coordinate"]
            CIGAR = reads_information_dict[assembly_id][read_id]["CIGAR"]
            md_z = reads_information_dict[assembly_id][read_id]["MD:Z"]
            errors_str, write_read = mark_errors(md_z, seqs_len)

            # Skip over reads containing unknown nucleotides
            if "N" in read or not write_read:
                continue

            # Generate read version with fixed subsitution errors (for quality check)
            seq_substitution_errors_fixed = apply_mutations(read, errors_str)

            # Proces sequences without indels
            if CIGAR == str(seqs_len) + "M":
                if seq_substitution_errors_fixed != assembly_seq[start_coord - 1 : start_coord + seqs_len - 1]:
                    print("Sequence was not back-substituted correctly. Skipping read.")
                    write_read = False
                    continue

                # Identify overlaps between CDS coordinates and genes; return for each CDS
                cds_overlaps = get_position_gene_overlaps(cds_coords_assembly, start_coord, seqs_len)

                # Get amino acid sequences and labels from each translated reading frame
                for rf in ["RF0", "RF1", "RF2"]:
                    # Translate reading frame 0
                    if rf == "RF0":
                        # Extract amino acid sequence and labels for nucleotide-level and amino-acid level labels
                        rf0_labels, rf0_labels_nt = generate_rf_labels(start_coord, seqs_len_rf0, cds_overlaps)
                        rf0_seq = encode_nucleotide_to_amino_acid(read)
                        rf0_labels = rf0_labels

                        assert len(rf0_labels_nt) == seqs_len_rf0

                        # Extract all coding fragments for quality check
                        if sum(rf0_labels) != 0:
                            correct_seq = encode_nucleotide_to_amino_acid(seq_substitution_errors_fixed)
                            coding_seqs_aa_rf = extract_coding_sequences(correct_seq, rf0_labels)
                            coding_seqs_all_read += coding_seqs_aa_rf

                    # Translate reading frame 1 (starts 1 position within nucleotide sequence)
                    if rf == "RF1":
                        rf1_labels, rf1_labels_nt = generate_rf_labels(start_coord + 1, seqs_len_rf1, cds_overlaps)
                        rf1_seq = encode_nucleotide_to_amino_acid(read[1:])

                        assert len(rf1_labels_nt) == seqs_len_rf1, print(len(rf1_labels_nt), seqs_len_rf1)

                        if sum(rf1_labels) != 0:
                            correct_seq = encode_nucleotide_to_amino_acid(seq_substitution_errors_fixed[1:])
                            coding_seqs_aa_rf = extract_coding_sequences(correct_seq, rf1_labels)
                            coding_seqs_all_read += coding_seqs_aa_rf

                    # Translate reading frame 2 (starts 2 positions within nucleotide sequence)
                    if rf == "RF2":
                        rf2_labels, rf2_labels_nt = generate_rf_labels(start_coord + 2, seqs_len_rf2, cds_overlaps)
                        rf2_seq = encode_nucleotide_to_amino_acid(read[2:])

                        assert len(rf2_labels_nt) == seqs_len_rf2

                        if sum(rf2_labels) != 0:
                            correct_seq = encode_nucleotide_to_amino_acid(seq_substitution_errors_fixed[2:])
                            coding_seqs_aa_rf = extract_coding_sequences(correct_seq, rf2_labels)
                            coding_seqs_all_read += coding_seqs_aa_rf

                write_read = quality_check_CDS_fragments(coding_seqs_all_read, accession)

            # Write read information if all quality checks are passed
            if write_read:
                # Get CDS coordinates on read and connections between CDS fragments (if any)
                cds_overlaps_read = mark_intervals(rf0_labels_nt, rf1_labels_nt, rf2_labels_nt, rf0_labels, rf1_labels, rf2_labels)

                if write_read:
                    reads_correct += 1
                    row_data = {
                        "read_name": read_id,
                        "read": read,
                        "cds_coords": cds_overlaps_read,
                        "cds_fragments_connection": None,
                        "start_coord": start_coord,
                        "assembly_id": assembly_id,
                        "CIGAR": CIGAR,
                        "MD:Z": md_z,
                        "indel_positions": None,
                        "accession": accession,
                        "rf0_seq": rf0_seq,
                        "rf0_labels": rf0_labels,
                        "rf1_seq": rf1_seq,
                        "rf1_labels": rf1_labels,
                        "rf2_seq": rf2_seq,
                        "rf2_labels": rf2_labels,
                        "strand": strand,
                    }

                    # Append the row to output df
                    processed_reads_df = pd.concat([processed_reads_df, pd.DataFrame([row_data])], ignore_index=True)

            if not write_read:
                reads_wrong += 1

    except KeyError:
        print("No reads on assembly.")

    return processed_reads_df, reads_correct, reads_wrong


###Main###
def run_pipeline(seqs_len, accession, partition, error_rates):
    """
    Run entire dataset creation pipeline (integrates all functions defined above).

    Args:
        seqs_len (int): Read length
        accession (str): RefSeq accession number

    Outputs: csv-file and fasta-file with labelled reads and additional information attributes.

    """

    # Get dicts with annotated CDS intervals for both strands (_uncertain_ marks annotations we are not sure of)
    cds_coords_template, cds_coords_uncertain_template, cds_coords_complement, cds_coords_uncertain_complement = extract_CDS(accession)

    # Extract all simulated reads and their information attributes; remove reads overlapping with aeras where we are not sure of the CDS annotations
    reads_information_dict_template = proces_reads_to_dict(accession, seqs_len, cds_coords_uncertain_template, "template_strand", partition, error_rates)
    reads_information_dict_complement = proces_reads_to_dict(accession, seqs_len, cds_coords_uncertain_complement, "complement_strand", partition, error_rates)

    # Define dataset attributes
    attributes = [
        "read_name",
        "read",
        "cds_coords",
        "cds_fragments_connection",
        "start_coord",
        "assembly_id",
        "CIGAR",
        "MD:Z",
        "indel_positions",
        "accession",
        "rf0_seq",
        "rf0_labels",
        "rf1_seq",
        "rf1_labels",
        "rf2_seq",
        "rf2_labels",
        "strand",
    ]
    processed_reads_df = pd.DataFrame(columns=attributes)

    # Extract genome filename and path
    genome_files = os.listdir(f"{data_base_path}/data/raw_data/genome_data/{accession}/")
    genome_fasta_file = [file for file in genome_files if file.startswith(accession)][0]
    genome_filename = f"{data_base_path}/data/raw_data/genome_data/{accession}/{genome_fasta_file}"

    # Initialize counters
    reads_correct = 0
    reads_wrong = 0

    # Iterate through genome assembly sequences (chromosome(s), plasmid(s))
    for record in SeqIO.parse(genome_filename, "fasta"):
        assembly_id = record.id
        assembly_seq = record.seq
        assembly_seq_rev = record.seq.reverse_complement()

        processed_reads_df, reads_correct, reads_wrong = process_strand_reads(
            assembly_id, assembly_seq, accession, seqs_len, cds_coords_template, reads_information_dict_template, processed_reads_df, reads_correct, reads_wrong, "+"
        )
        processed_reads_df, reads_correct, reads_wrong = process_strand_reads(
            assembly_id, assembly_seq_rev, accession, seqs_len, cds_coords_complement, reads_information_dict_complement, processed_reads_df, reads_correct, reads_wrong, "-"
        )

    # check if directories exist, if not create them
    if not os.path.exists(f"{data_base_path}/data/processed_data/reads_processed/{partition}/{error_rates}/csv/"):
        os.makedirs(f"{data_base_path}/data/processed_data/reads_processed/{partition}/{error_rates}/csv/")
    if not os.path.exists(f"{data_base_path}/data/processed_data/reads_processed/{partition}/{error_rates}/fasta/"):
        os.makedirs(f"{data_base_path}/data/processed_data/reads_processed/{partition}/{error_rates}/fasta/")

    print(reads_correct, reads_wrong)

    if round(reads_correct / (reads_correct + reads_wrong) * 100, 2) < 100:
        print(f"Fraction of correct reads for {accession}: {round(reads_correct / (reads_correct + reads_wrong) * 100, 2)}%", flush=True)
    else:
        print("All sequences processed properly for: ", accession)

    # Save datasets as both csv and fasta files.
    processed_reads_df.to_csv(f"{data_base_path}/data/processed_data/reads_processed/{partition}/{error_rates}/csv/{accession}.csv.gz", compression="gzip")

    # Create fasta file
    with gzip.open(f"{data_base_path}/data/processed_data/reads_processed/{partition}/{error_rates}/fasta/{accession}.fasta.gz", "wt") as fasta_out:  # 'wt' = write text mode
        for _, row in processed_reads_df.iterrows():
            fasta_out.write(f">{row['read_name']}|{row['strand']}|{row['assembly_id']}|{row['cds_coords']}|{row['indel_positions']}\n{row['read']}\n")


# Run pipeline on X processes
def run_pipeline_wrapper(args):
    seqs_len, accession, partition, error_rates = args
    print(f"Processing {accession}", flush=True)
    run_pipeline(seqs_len, accession, partition, error_rates)


def main():
    args_list = [(seqs_len, accession, partition, error_rates) for accession in accessions]  ###Only process first accession for testing
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(run_pipeline_wrapper, args) for args in args_list]
        for future in as_completed(futures):
            future.result()


process_train_val_reads = False
process_test_reads = True

if __name__ == "__main__":
    if process_train_val_reads:
        partition = "train_val"
        accessions = accessions_train + accessions_val

        # Continue processing from where left off
        # accessions = set(accessions_train + accessions_val)
        # accessions_processed = os.listdir(f"{data_base_path}/data/processed_data/reads_processed/train_val/with_substitution_errors/csv/")
        # accessions_processed = set([accession[:-4] for accession in accessions_processed])
        # accessions = accessions - accessions_processed

        seqs_len = 300
        for error_rates in ["with_substitution_errors", "without_errors"]:
            print("======================================")
            print("Data partition: ", partition)
            print("Error rates: ", error_rates)
            print("Processing samples...")
            main()
            print("======================================")

    if process_test_reads:
        partition = "test"
        accessions = accessions_test

        for seqs_len in [30, 60, 75, 100, 150, 300, 700, 1000]:

            for error_rates in [f"without_errors_{str(seqs_len)}bp"]:
                print("======================================")
                print("Data partition: ", partition)
                print("Dataset: ", error_rates)
                print("Read length: ", seqs_len)
                print("Processing samples...")
                main()
                print("======================================")
