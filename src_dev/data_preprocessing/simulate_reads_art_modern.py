import gzip
import os
import re

from Bio import SeqIO
from tqdm import tqdm

compute_machine = "cluster"  # Options: "cluster" or "local"

if compute_machine == "cluster":
    data_base_path = "/tmp/nrt204/FragmentPredictor"
else:
    data_base_path = "../.."

accessions_test = open(f"{data_base_path}/data/processed_data/genome_partitions/test_partition_accessions.txt").read().splitlines()

#accessions_test = accessions_test[0:1]
#print(accessions_test)


def prepare_genome(accession):
    genome_files = os.listdir(f"{data_base_path}/data/raw_data/genome_data/{accession}/")
    genome_fasta_file = [file for file in genome_files if file.startswith(accession) and file.endswith((".fna"))][0]
    genome_filename = f"{data_base_path}/data/raw_data/genome_data/{accession}/{genome_fasta_file}"

    with open(f"./tmp/{accession}_reverse_complement.fasta", "w") as outfile_rev_comp:
        for record in SeqIO.parse(genome_filename, "fasta"):
            outfile_rev_comp.write(">" + str(record.description) + "\n")
            outfile_rev_comp.write(str(record.seq.reverse_complement()) + "\n")

    return genome_fasta_file


def simulate_reads(accession, read_length, coverage, error_profile) -> None:
    """
    Simulate reads with ART (HS25 profile) for a given accession and read length.
    Simulates on both the template and complement strand separately.

    Args:
        accession (str): The accession number of the genome.
        read_length (int): The length of the reads to simulate.
        coverage (float): The target coverage per strand.
        error_profile (str): The ART error profile to use (e.g., "HS25").

    Returns:
        Returns simulated reads (/reads/filename.fasta.gz) and alignments (/alignments/filename.sam)
        for both DNA strands.
    """

    # Create needed directories if they do not exist
    base = f"{data_base_path}/data/processed_data/simulated_reads/test/{error_profile}"

    subdirs = [
        "template_strand/reads",
        "template_strand/alignments",
        "complement_strand/reads",
        "complement_strand/alignments",
    ]

    for sub in subdirs:
        os.makedirs(os.path.join(base, sub), exist_ok=True)

    genome_fasta_file = prepare_genome(accession)

    strands = [
        ("template_strand", f"{data_base_path}/data/raw_data/genome_data/{accession}/{genome_fasta_file}"),
        ("complement_strand", f"./tmp/{accession}_reverse_complement.fasta"),
    ]

    for strand_dir, ref in strands:
        strand = "template" if strand_dir == "template_strand" else "complement"
        tmp = f"./tmp/{accession}_{strand_dir}"

        # ART simulates from both strands of the reference, so double the coverage
        # so that after filtering to forward-mapped reads only we achieve the target coverage
        os.system(
            f"art_modern --mode wgs --lc se --i-seed 42 --i-type fasta --i-file {ref} --i-fcov {coverage * 2} "
            f"--o-fasta {tmp}.fasta.gz --o-fasta-compression gzip --o-sam {base}/{strand_dir}/alignments/{accession}_alignments.bam --o-sam-use_m --o-sam-write_bam --read_len {read_length} "
            f"--builtin_qual_file {error_profile}"
        )

        #Filter out reads mapping to the reverse strand (flag 16) or unmapped (flag 4), and sort BAM by coordinate for downstream processing
        os.system(f"samtools view -F 16 -bS {base}/{strand_dir}/alignments/{accession}_alignments.bam | samtools sort -o {base}/{strand_dir}/alignments/{accession}_alignments.bam")

        #Produce a fasta file with the reads only for the reference strand
        os.system(f"samtools fasta {base}/{strand_dir}/alignments/{accession}_alignments.bam | gzip > {base}/{strand_dir}/reads/{accession}_simulated_reads.fasta.gz")

    # Clean up tmp files
    for f in os.listdir("./tmp"):
        if f.startswith(accession):
            os.remove(os.path.join("./tmp", f))


def simulate_test_reads(accessions_test, read_length, error_profile) -> None:
    """
    Simulate reads for all genomes in the test partition using a built-in ART error profile.

    Args:
        accessions_test (list): List of accession numbers for genomes in the test partition.
        read_length (int): The length of the reads to simulate.
        error_profile (str): The ART error profile to use (e.g., "HS25").
    """
    for accession in tqdm(accessions_test, desc="Simulating reads for test partition"):
        print(f"=== Simulating reads for {accession} ===", flush=True)
        simulate_reads(accession, read_length=read_length, coverage=1, error_profile=error_profile)

    print("\nRead simulation complete.")


if __name__ == "__main__":
    print("Simulating HiSeq2500 reads...", flush = True)
    simulate_test_reads(accessions_test, read_length=150, error_profile="HiSeq2500_150bp")
    print("Simulating NextSeq500 reads...", flush = True)
    simulate_test_reads(accessions_test, read_length=150, error_profile="NextSeq500_150bp")
    print("Simulating MiSeq v3 reads...", flush = True)
    simulate_test_reads(accessions_test, read_length=300, error_profile="MiSeq_v3_300bp")
