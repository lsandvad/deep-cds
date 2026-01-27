import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(['science', 'nature'])


def count_substitutions_in_mdz(mdz_string):
    """
    Count substitution errors in an MD:Z tag.
    Substitutions are indicated by A, G, C, T (not preceded by ^).
    The ^ symbol indicates deletions in the reference, which we skip here.
    """
    if pd.isna(mdz_string):
        return 0

    # Remove deletion markers (^[ACGT]+) first, then count remaining bases
    # Deletions are marked as ^X where X is one or more bases
    cleaned = re.sub(r'\^[ACGT]+', '', str(mdz_string))

    # Count remaining A, G, C, T characters (these are substitutions)
    substitutions = len(re.findall(r'[ACGT]', cleaned))
    return substitutions


def count_indels_in_position(indel_string):
    """
    Count indel errors from indel_positions column.
    Format: position followed by I (insertion) or D (deletion), e.g., '482D', '811I'
    Multiple indels can be separated by semicolons or other delimiters.
    """
    if pd.isna(indel_string):
        return 0

    # Count occurrences of I or D
    indels = len(re.findall(r'[ID]', str(indel_string)))
    return indels


def get_read_length_from_cigar(cigar_string):
    """
    Calculate the read length from a CIGAR string.
    M (match/mismatch), I (insertion), S (soft clip) consume query bases.
    """
    if pd.isna(cigar_string):
        return 0

    # Sum up M, I, S, = (sequence match), X (sequence mismatch) operations
    operations = re.findall(r'(\d+)([MISX=])', str(cigar_string))
    length = sum(int(count) for count, op in operations)
    return length


def extract_error_rates_from_dirname(dirname):
    """
    Extract indel and substitution rates from directory name.
    Example: 'with_errors_1.25e-05i_0.01s_1000bp' -> (1.25e-05, 0.01)
    """
    match = re.search(r'([\d.e-]+)i_([\d.e-]+)s', dirname)
    if match:
        indel_rate = float(match.group(1))
        sub_rate = float(match.group(2))
        return indel_rate, sub_rate
    return None, None


def analyze_all_csv_files():
    """
    Analyze all CSV files across test data directories with different error rates.
    """
    base_path = "../../data/processed_data/reads_processed/test/"
    all_dirs = os.listdir(base_path)

    # Filter directories: start with "with_errors" and end with "1000bp"
    test_data_dirs = [d for d in all_dirs
                      if d.startswith("with_errors") and d.endswith("1000bp")]
    test_data_dirs = sorted(test_data_dirs)

    print(f"Found {len(test_data_dirs)} error rate conditions:")
    for d in test_data_dirs:
        print(f"  - {d}")

    # Colorblind-friendly palette (Tol's bright)
    colors = ['#44AA99', '#AA4499', '#DDCC77', '#88CCEE', '#CC6677']

    # Store data per error rate condition
    data_by_condition = {}

    for test_data_dir in test_data_dirs:
        csv_dir = f"{base_path}{test_data_dir}/csv/"
        if not os.path.exists(csv_dir):
            continue

        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv') or f.endswith('.csv.gz')]

        substitutions_per_read = []
        indels_per_read = []
        read_lengths = []

        for csv_file in csv_files:
            filepath = f"{csv_dir}{csv_file}"

            if csv_file.endswith('.gz'):
                df = pd.read_csv(filepath, compression="gzip")
            else:
                df = pd.read_csv(filepath)

            subs_per_read = df["MD:Z"].apply(count_substitutions_in_mdz)
            substitutions_per_read.extend(subs_per_read.tolist())

            indels_pr = df["indel_positions"].apply(count_indels_in_position)
            indels_per_read.extend(indels_pr.tolist())

            lengths = df["CIGAR"].apply(get_read_length_from_cigar)
            read_lengths.extend(lengths.tolist())

        # Convert to numpy arrays
        substitutions_per_read = np.array(substitutions_per_read)
        indels_per_read = np.array(indels_per_read)
        read_lengths = np.array(read_lengths)

        # Calculate rates
        total_bases = read_lengths.sum()
        sub_rate = substitutions_per_read.sum() / total_bases if total_bases > 0 else 0
        indel_rate = indels_per_read.sum() / total_bases if total_bases > 0 else 0

        # Extract nominal rates from dirname for labeling
        nominal_indel, nominal_sub = extract_error_rates_from_dirname(test_data_dir)

        data_by_condition[test_data_dir] = {
            'substitutions': substitutions_per_read,
            'indels': indels_per_read,
            'read_lengths': read_lengths,
            'empirical_sub_rate': sub_rate,
            'empirical_indel_rate': indel_rate,
            'nominal_sub_rate': nominal_sub,
            'nominal_indel_rate': nominal_indel,
            'total_reads': len(substitutions_per_read)
        }

        print(f"\nProcessed {test_data_dir}:")
        print(f"  Reads: {len(substitutions_per_read):,}")
        print(f"  Nominal sub rate: {nominal_sub}, Empirical: {sub_rate:.6f}")
        print(f"  Nominal indel rate: {nominal_indel}, Empirical: {indel_rate:.6f}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS BY CONDITION")
    print("="*60)
    for condition, data in data_by_condition.items():
        print(f"\n{condition}:")
        print(f"  Total reads: {data['total_reads']:,}")
        print(f"  Substitutions - Mean: {data['substitutions'].mean():.2f}, Max: {data['substitutions'].max()}")
        print(f"  Indels - Mean: {data['indels'].mean():.4f}, Max: {data['indels'].max()}")
    print("="*60)

    # Create plots with overlaid histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})

    # Find global max for consistent binning
    all_subs = np.concatenate([d['substitutions'] for d in data_by_condition.values()])
    all_indels = np.concatenate([d['indels'] for d in data_by_condition.values()])
    max_subs = int(all_subs.max())
    max_indels = int(all_indels.max())

    bins_subs = np.arange(0, max_subs + 2) - 0.5
    bins_indels = np.arange(0, max_indels + 2) - 0.5

    # Plot 1: Substitution errors distribution (overlaid)
    ax1 = axes[0]
    for i, (condition, data) in enumerate(data_by_condition.items()):
        label = f"Sub rate {data['nominal_sub_rate']} (n={data['total_reads']:,})"
        ax1.hist(data['substitutions'], bins=bins_subs, alpha=0.5, color=colors[i % len(colors)],
                 label=label, edgecolor='none')

    ax1.set_xlabel('Number of Substitution Errors per Read')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Substitution Errors per Read')
    ax1.legend()

    # Plot 2: Indel errors distribution (overlaid, log scale)
    ax2 = axes[1]
    for i, (condition, data) in enumerate(data_by_condition.items()):
        label = f"Indel rate {data['nominal_indel_rate']} (n={data['total_reads']:,})"
        ax2.hist(data['indels'], bins=bins_indels, alpha=0.5, color=colors[i % len(colors)],
                 label=label, edgecolor='none')

    ax2.set_yscale('log')
    ax2.set_xlabel('Number of Indel Errors per Read')
    ax2.set_ylabel('Frequency (log scale)')
    ax2.set_title('Distribution of Indel Errors per Read')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('error_distribution_histograms.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nPlot saved to: error_distribution_histograms.png")

    return data_by_condition


if __name__ == "__main__":
    results = analyze_all_csv_files()



"""
accessions_train = open("../../data/processed_data/genome_partitions/train_partition_accessions.txt").read().splitlines()
accessions_val = open("../../data/processed_data/genome_partitions/val_partition_accessions.txt").read().splitlines()
accessions_test = open("../../data/processed_data/genome_partitions/test_partition_accessions.txt").read().splitlines()


####CHECKS TO MAKE: 
#1. Number of reads corresponds to coverage of approx. 1
#2. Sequence errors in the reads correspond to the specified error rates

partition = "test"
error_rates = "with_errors_5e-06i_0.004s_60bp"
strand = "template_strand"
accession = accessions_test[0]  #Example accession to check

count_reads = 0

#Open Mason output information (.bam-file)
with pysam.AlignmentFile(f"../../data/processed_data/simulated_reads/{partition}/{error_rates}/{strand}/alignments/{accession}_alignments.bam", "rb") as bam_infile:
    for read in bam_infile:
        count_reads += 1
        #Extract information from the read 
        start_coordinate = read.reference_start + 1  #pysam is 0-based, convert to 1-base
        assembly = read.reference_name
            
        #Get read-specific information
        read_id = read.query_name
        CIGAR = read.cigarstring
        read_seq = read.query_sequence
        MD_Z = read.get_tag("MD") if read.has_tag("MD") else None  # Handle missing MD tag

print(f"Total length for {accession} in {strand}: {count_reads * 60}")
"""