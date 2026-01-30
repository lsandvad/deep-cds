import os

import numpy as np
from Bio import SeqIO
from tqdm import tqdm

accessions_train = open("../../data/processed_data/genome_partitions/train_partition_accessions.txt").read().splitlines()
accessions_val = open("../../data/processed_data/genome_partitions/val_partition_accessions.txt").read().splitlines()
accessions_test = open("../../data/processed_data/genome_partitions/test_partition_accessions.txt").read().splitlines()


def get_number_of_reads_to_simulate(accession, read_length, coverage):
    """
    Get the number of reads to simulate for a given accession and read length.

    Args:
        accession (str): The accession number of the genome.
        read_length (int): The length of the reads to simulate.
        coverage (float): The coverage to simulate.

    Returns:
        num_reads (int): number of reads to simulate.
        genome_fasta_file (str): relative filepath and name of genomic fasta file.
        outfile_rev_comp (fasta file): write a fasta file with the reverse complement genome sequence.
    """

    # Read the genome file
    genome_files = os.listdir(f"../../data/raw_data/genome_data/{accession}/")
    genome_fasta_file = [file for file in genome_files if file.startswith(accession)][0]
    genome_filename = f"../../data/raw_data/genome_data/{accession}/{genome_fasta_file}"

    # Initialize
    outfile_rev_comp = open(accession + "_reverse_complement.fasta", "w")
    genome_len = 0

    # Loop over sequences in the genome file
    for record in SeqIO.parse(genome_filename, "fasta"):
        # Store length of full genomic sequence (chromosome(s) + plasmids)
        genome_len += len(record.seq)

        # Get reverse complemented sequences
        assembly_description = record.description
        assembly_seq = record.seq
        rev_comp_seq = assembly_seq.reverse_complement()

        # Write to outfile
        outfile_rev_comp.write(">" + str(assembly_description) + "\n")
        outfile_rev_comp.write(str(rev_comp_seq) + "\n")

    # Calculate the number of reads to simulate (round up to nearest integer)
    num_reads = int(np.ceil((genome_len * coverage) / read_length))

    outfile_rev_comp.close()

    return num_reads, genome_fasta_file


def simulate_reads(accession, read_length, coverage, substitution_error_rate, indel_error_rates, with_errors, genome_partition, fragment_mean_size=None) -> None:
    """
    Simulate reads with Mason for a given accession and read length. Simulates on both the template and complement strand.

    Args:
        accession (str): The accession number of the genome.
        read_length (int): The length of the reads to simulate.
        coverage (float): The coverage to simulate.
        substitution_error_rate (float): The substitution error rate to simulate.
        indel_error_rates (float): The indel error rate to simulate.
        with_errors (str or bool): Whether to simulate with errors ("indels_substitutions", "substitutions_only", or False).
        genome_partition (str): The genome partition ("train_val" or "test").
        fragment_mean_size (int, optional): The mean fragment size. Defaults to None, which sets it to read_length + 100.

    Returns:
        Returns simulated reads (/reads/filename.fasta.gz) and alignments (/alignments/filename.bam) with reads, sequencing error information etc., for both DNA strands.
    """

    if with_errors == "indels_substitutions":
        if genome_partition == "test":
            error_rates_str = f"with_errors_{str(indel_error_rates)}i_{str(substitution_error_rate)}s_{str(read_length)}bp"
        else:
            error_rates_str = "with_errors"
    elif with_errors == "substitutions_only":
        if genome_partition == "test":
            error_rates_str = f"with_errors_0i_{str(substitution_error_rate)}s_{str(read_length)}bp"
        else:
            error_rates_str = "with_substitution_errors"
    else:
        if genome_partition == "test":
            error_rates_str = f"without_errors_{str(read_length)}bp"
        else:
            error_rates_str = "without_errors"

    # If we dont pre-define fragment mean size, set it to read_length + 100
    if fragment_mean_size is None:
        fragment_mean_size = read_length + 100

    # Create needed directories if they do not exist
    base = f"../../data/processed_data/simulated_reads/{genome_partition}/{error_rates_str}"

    subdirs = [
        "template_strand/reads",
        "template_strand/alignments",
        "complement_strand/reads",
        "complement_strand/alignments",
    ]

    for sub in subdirs:
        os.makedirs(os.path.join(base, sub), exist_ok=True)

    num_reads, genome_fasta_file = get_number_of_reads_to_simulate(accession=accession, read_length=read_length, coverage=coverage)

    # Run Mason to simulate reads
    if with_errors:
        # Simulate reads for template strand
        os.system(
            f"mason_simulator -ir ../../data/raw_data/genome_data/{accession}/{genome_fasta_file} -n {num_reads} --illumina-read-length {read_length} \
                --illumina-prob-insert {indel_error_rates} --illumina-prob-deletion {indel_error_rates} --illumina-prob-mismatch {substitution_error_rate} \
                -o {base}/template_strand/reads/{accession}_simulated_reads.fasta.gz -oa {base}/template_strand/alignments/{accession}_alignments.bam \
                --fragment-mean-size {fragment_mean_size} --seed 42  --seq-strands forward --read-name-prefix {accession}_simulated_reads_template"
        )

        # Remove unnecessary files
        if os.path.exists(f"../../data/raw_data/genome_data/{accession}/{genome_fasta_file}.fai"):
            os.remove(f"../../data/raw_data/genome_data/{accession}/{genome_fasta_file}.fai")

        # Simulate reads for complement strand
        os.system(
            f"mason_simulator -ir {accession}_reverse_complement.fasta -n {num_reads} --illumina-read-length {read_length} --illumina-prob-insert {indel_error_rates} \
                --illumina-prob-deletion {indel_error_rates} --illumina-prob-mismatch {substitution_error_rate} \
                -o {base}/complement_strand/reads/{accession}_simulated_reads.fasta.gz -oa {base}/complement_strand/alignments/{accession}_alignments.bam \
                --fragment-mean-size {fragment_mean_size} --seed 42  --seq-strands forward --read-name-prefix {accession}_simulated_reads_complement"
        )

    else:
        # Run Mason to simulate reads
        os.system(
            f"mason_simulator -ir ../../data/raw_data/genome_data/{accession}/{genome_fasta_file} -n {num_reads} --illumina-read-length {read_length} \
                --illumina-prob-insert 0.0 --illumina-prob-deletion 0.0  --illumina-prob-mismatch 0.0 \
                --illumina-prob-mismatch-begin 0.0 --illumina-prob-mismatch-end 0.0 \
                -o {base}/template_strand/reads/{accession}_simulated_reads.fasta.gz -oa {base}/template_strand/alignments/{accession}_alignments.bam \
                --fragment-mean-size {fragment_mean_size} --seed 42  --seq-strands forward --read-name-prefix {accession}_simulated_reads_template"
        )

        # Remove unnecessary files
        if os.path.exists(f"../../data/raw_data/genome_data/{accession}/{genome_fasta_file}.fai"):
            os.remove(f"../../data/raw_data/genome_data/{accession}/{genome_fasta_file}.fai")

        # Simulate reads for complement strand
        os.system(
            f"mason_simulator -ir {accession}_reverse_complement.fasta -n {num_reads} --illumina-read-length {read_length} \
                --illumina-prob-insert 0.0 --illumina-prob-deletion 0.0  --illumina-prob-mismatch 0.0 \
                --illumina-prob-mismatch-begin 0.0 --illumina-prob-mismatch-end 0.0 \
                -o {base}/complement_strand/reads/{accession}_simulated_reads.fasta.gz -oa {base}/complement_strand/alignments/{accession}_alignments.bam \
                --fragment-mean-size {fragment_mean_size} --seed 42  --seq-strands forward --read-name-prefix {accession}_simulated_reads_complement"
        )

    # Remove unnecessary files
    if os.path.exists(f"{accession}_reverse_complement.fasta.fai"):
        os.remove(f"{accession}_reverse_complement.fasta.fai")

    if os.path.exists(f"{accession}_reverse_complement.fasta"):
        os.remove(f"{accession}_reverse_complement.fasta")


# Run simulation pipeline with Mason for all genomes used for model development
def simulate_training_validation_reads(accessions_train, accessions_val) -> None:
    """
    Simulate reads for all genomes in the training and validation partitions.
    Simulate three datasets:
        1) with both indel and substitution errors
        2) with only substitution errors
        3) without sequencing errors
    """
    for accession in tqdm(accessions_train + accessions_val):
        # Simulate reads with sequencing errors (simulate 0.05 % indel error rate, and 0.5 % substitution error rate)
        simulate_reads(accession, 
                       read_length=300, 
                       coverage=1, 
                       substitution_error_rate=0.005, 
                       indel_error_rates=0.00025, 
                       with_errors="indels_substitutions", 
                       genome_partition="train_val")

        # Dataset only with substitution errors (0.5 % substitution error rate, no indel errors)
        simulate_reads(accession, 
                       read_length=300, 
                       coverage=1, 
                       substitution_error_rate=0.005, 
                       indel_error_rates=0.0, 
                       with_errors="substitutions_only", 
                       genome_partition="train_val")

        # Simulate reads without sequencing errors
        simulate_reads(accession, 
                       read_length=300, 
                       coverage=1, 
                       substitution_error_rate=0, 
                       indel_error_rates=0, 
                       with_errors=False, 
                       genome_partition="train_val")

    print("Read simulation complete.")


def simulate_test_reads(accessions_test, read_length, fragment_mean_size=None) -> None:
    """
    Simulate reads for all genomes in the test partition.
    Simulate four datasets:
        1) with both indel and substitution errors (realistic error rates)
        2) with both indel and substitution errors (higher error rates)
        3) with both indel and substitution errors (even higher error rates)
        4) without sequencing errors

    Args:
        accessions_test (list): List of accession numbers for genomes in the test partition.
        read_length (int): The length of the reads to simulate.

    """
    ##Simulate reads for test
    for accession in accessions_test:
        # Simulate reads with sequencing errors (0.001% indels, 0.4% substituion errors)
        simulate_reads(
            accession,
            read_length=read_length,
            coverage=1,
            substitution_error_rate=0.004,
            indel_error_rates=0.000005,
            with_errors="indels_substitutions",
            genome_partition="test",
            fragment_mean_size=fragment_mean_size,
        )

        # Simulate reads with sequencing errors (0.0025% indels, 1% substituion errors)
        simulate_reads(
            accession,
            read_length=read_length,
            coverage=1,
            substitution_error_rate=0.01,
            indel_error_rates=0.0000125,
            with_errors="indels_substitutions",
            genome_partition="test",
            fragment_mean_size=fragment_mean_size,
        )

        # Simulate reads with sequencing errors (0.0075% indels, 3% substituion errors)
        simulate_reads(
            accession,
            read_length=read_length,
            coverage=1,
            substitution_error_rate=0.03,
            indel_error_rates=0.0000375,
            with_errors="indels_substitutions",
            genome_partition="test",
            fragment_mean_size=fragment_mean_size,
        )

        # Simulate reads without sequencing errors (0% indels, 0% substituion errors)
        simulate_reads(
            accession, 
            read_length=read_length, 
            coverage=1, 
            substitution_error_rate=0, 
            indel_error_rates=0,
            with_errors=False, 
            genome_partition="test", 
            fragment_mean_size=fragment_mean_size
        )

    print("Read simulation complete.")


if __name__ == "__main__":
    # Simulate reads for training and validation genomes
    simulate_training_validation_reads(accessions_train, accessions_val)

    # Simulate reads for test genomes
    for read_length in [60, 75, 100, 150, 300]:
        print(f"Simulating test reads with length {read_length}")
        simulate_test_reads(accessions_test, read_length)

    # Simulate reads for test genomes
    for read_length in [700, 1000]:
        print(f"Simulating test reads with length {read_length}")
        simulate_test_reads(accessions_test, read_length, fragment_mean_size=read_length + 500)