import os
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
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

    colors = ['#44AA99', '#AA4499', '#63A31A', '#88CCEE', '#CC6677']

    # Read lengths to analyze
    read_lengths_to_analyze = ['60bp', '75bp', '100bp', '150bp', '300bp']

    all_results = {}

    for read_length in read_lengths_to_analyze:
        print(f"\n{'='*60}")
        print(f"ANALYZING READ LENGTH: {read_length}")
        print("="*60)

        # Filter directories for this read length
        test_data_dirs = [d for d in all_dirs
                          if d.startswith("with_errors") and d.endswith(read_length)]
        test_data_dirs = sorted(test_data_dirs)

        if not test_data_dirs:
            print(f"No directories found for {read_length}, skipping...")
            continue

        print(f"Found {len(test_data_dirs)} error rate conditions:")
        for d in test_data_dirs:
            print(f"  - {d}")

        # Store data per error rate condition
        data_by_condition = {}

        for test_data_dir in test_data_dirs:
            csv_dir = f"{base_path}{test_data_dir}/csv/"
            if not os.path.exists(csv_dir):
                continue

            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv') or f.endswith('.csv.gz')]
            #csv_files = csv_files[0:5] #for testing how plot looks without the high runtime
            
            substitutions_per_read = []
            indels_per_read = []
            read_lengths_list = []

            for csv_file in csv_files:
                filepath = f"{csv_dir}{csv_file}"

                if csv_file.endswith('.gz'):
                    df = pd.read_csv(filepath, compression="gzip", low_memory=False)
                else:
                    df = pd.read_csv(filepath, low_memory=False)

                subs_per_read = df["MD:Z"].apply(count_substitutions_in_mdz)
                substitutions_per_read.extend(subs_per_read.tolist())

                indels_pr = df["indel_positions"].apply(count_indels_in_position)
                indels_per_read.extend(indels_pr.tolist())

                lengths = df["CIGAR"].apply(get_read_length_from_cigar)
                read_lengths_list.extend(lengths.tolist())

            # Convert to numpy arrays
            substitutions_per_read = np.array(substitutions_per_read)
            indels_per_read = np.array(indels_per_read)
            read_lengths_list = np.array(read_lengths_list)

            # Calculate rates
            total_bases = read_lengths_list.sum()
            sub_rate = substitutions_per_read.sum() / total_bases if total_bases > 0 else 0
            indel_rate = indels_per_read.sum() / total_bases if total_bases > 0 else 0

            # Extract nominal rates from dirname for labeling
            nominal_indel, nominal_sub = extract_error_rates_from_dirname(test_data_dir)

            data_by_condition[test_data_dir] = {
                'substitutions': substitutions_per_read,
                'indels': indels_per_read,
                'read_lengths': read_lengths_list,
                'empirical_sub_rate': sub_rate,
                'empirical_indel_rate': indel_rate,
                'nominal_sub_rate': nominal_sub,
                'nominal_indel_rate': nominal_indel,
                'total_reads': len(substitutions_per_read)
            }

            print(f"\nProcessed {test_data_dir}:")
            print(f"  Reads: {len(substitutions_per_read):,}")
            if nominal_sub is not None and nominal_indel is not None:
                print(f"  Nominal sub rate: {nominal_sub * 100:.4f}%, Empirical: {sub_rate * 100:.4f}%")
                print(f"  Nominal indel rate: {nominal_indel * 2 * 100:.6f}%, Empirical: {indel_rate * 100:.6f}%")
            else:
                print(f"  Empirical sub rate: {sub_rate * 100:.4f}%")
                print(f"  Empirical indel rate: {indel_rate * 100:.6f}%")

        # Print summary
        print("\n" + "-"*40)
        print(f"SUMMARY FOR {read_length}")
        print("-"*40)
        for condition, data in data_by_condition.items():
            print(f"\n{condition}:")
            print(f"  Total reads: {data['total_reads']:,}")
            print(f"  Substitutions - Mean: {data['substitutions'].mean():.2f}, Max: {data['substitutions'].max()}")
            print(f"  Indels - Mean: {data['indels'].mean():.4f}, Max: {data['indels'].max()}")

        # Create plots with overlaid histograms
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [2, 1]})
        fig.suptitle(f'Read length: {read_length}', fontsize=11, y=0.98)

        # Find global max for consistent binning
        all_subs = np.concatenate([d['substitutions'] for d in data_by_condition.values()])
        all_indels = np.concatenate([d['indels'] for d in data_by_condition.values()])
        max_subs = int(all_subs.max())
        max_indels = int(all_indels.max())

        bins_subs = np.arange(0, max_subs + 2) - 0.5

        # Sort conditions by nominal substitution rate (lowest first)
        sorted_conditions = sorted(data_by_condition.items(),
                                   key=lambda x: (x[1]['nominal_sub_rate'] or 0))

        # Plot 1: Substitution errors distribution (overlaid)
        ax1 = axes[0]
        for i, (condition, data) in enumerate(sorted_conditions):
            nominal_pct = data['nominal_sub_rate'] * 100 if data['nominal_sub_rate'] else 0
            empirical_pct = data['empirical_sub_rate'] * 100
            label = f"{nominal_pct:.2f}\\% (empirical = {empirical_pct:.2f}\\%)"
            ax1.hist(data['substitutions'], bins=bins_subs, alpha=0.5, color=colors[i % len(colors)],
                     label=label, edgecolor='none')

        ax1.set_xlabel('Number of Substitution Errors per Read', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_xlim(left=-0.5)  # Show full bar at 0
        # Format y-axis with scientific notation directly on tick labels
        def sci_formatter(x, pos):
            if x == 0:
                return '0'
            exp = int(np.floor(np.log10(abs(x))))
            coef = x / 10**exp
            if coef == 1:
                return f'$10^{{{exp}}}$'
            return f'${coef:.0f}\\times10^{{{exp}}}$'
        ax1.yaxis.set_major_formatter(FuncFormatter(sci_formatter))
        ax1.tick_params(labelsize=11)
        ax1.legend(title='Substitution error rate', fontsize=10, title_fontsize=10)

        # Plot 2: Indel errors - Grouped bar chart
        ax2 = axes[1]
        indel_values = np.arange(0, max_indels + 1)
        n_conditions = len(sorted_conditions)
        bar_width = 0.8 / n_conditions

        for i, (condition, data) in enumerate(sorted_conditions):
            nominal_pct = data['nominal_indel_rate'] * 2 * 100 if data['nominal_indel_rate'] else 0
            empirical_pct = data['empirical_indel_rate'] * 100
            label = f"{nominal_pct:.4f}\\% (empirical = {empirical_pct:.4f}\\%)"
            # Count occurrences of each indel value
            counts = [np.sum(data['indels'] == v) for v in indel_values]
            x_positions = indel_values + (i - n_conditions/2 + 0.5) * bar_width
            ax2.bar(x_positions, counts, width=bar_width, color=colors[i % len(colors)],
                    label=label, alpha=0.8)

        ax2.set_yscale('log')
        ax2.set_xlabel('Number of Indel Errors per Read', fontsize=12)
        ax2.set_ylabel('Frequency (log scale)', fontsize=12)
        ax2.set_xticks(indel_values)
        ax2.tick_params(labelsize=11)
        ax2.legend(title='Indel error rate', fontsize=10, title_fontsize=10)

        plt.tight_layout()
        os.makedirs("../../illustrations/testset_error_rates_plots", exist_ok=True)
        output_filename = f'../../illustrations/testset_error_rates_plots/error_distribution_histograms_{read_length}.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved to: {output_filename}")

        # Store results for this read length
        all_results[read_length] = data_by_condition

    return all_results


def analyze_art_modern_testsets():
    """
    Analyze CSV files from test sets simulated with art_modern using
    platform-specific quality profiles (HiSeq2500, NextSeq500, MiSeq v3)
    and produce a single overlaid plot comparing their error distributions.
    """
    base_path = "../../data/processed_data/reads_processed/test/"

    art_modern_dirs = {
        "HiSeq2500_150bp": "HiSeq2500 (150bp)",
        "NextSeq500_150bp": "NextSeq500 (150bp)",
        "MiSeq_v3_300bp":   "MiSeq v3 (300bp)",
    }

    colors = ['#44AA99', '#AA4499', '#63A31A']

    data_by_profile = {}

    for dirname, label in art_modern_dirs.items():
        csv_dir = f"{base_path}{dirname}/csv/"
        if not os.path.exists(csv_dir):
            print(f"Directory not found, skipping: {csv_dir}")
            continue

        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv') or f.endswith('.csv.gz')]

        substitutions_per_read = []
        indels_per_read = []
        read_lengths_list = []

        for csv_file in csv_files:
            filepath = f"{csv_dir}{csv_file}"
            if csv_file.endswith('.gz'):
                df = pd.read_csv(filepath, compression="gzip", low_memory=False)
            else:
                df = pd.read_csv(filepath, low_memory=False)

            substitutions_per_read.extend(df["MD:Z"].apply(count_substitutions_in_mdz).tolist())
            indels_per_read.extend(df["indel_positions"].apply(count_indels_in_position).tolist())
            read_lengths_list.extend(df["CIGAR"].apply(get_read_length_from_cigar).tolist())

        substitutions_per_read = np.array(substitutions_per_read)
        indels_per_read = np.array(indels_per_read)
        read_lengths_list = np.array(read_lengths_list)

        total_bases = read_lengths_list.sum()
        sub_rate = substitutions_per_read.sum() / total_bases if total_bases > 0 else 0
        indel_rate = indels_per_read.sum() / total_bases if total_bases > 0 else 0

        data_by_profile[label] = {
            'substitutions': substitutions_per_read,
            'indels': indels_per_read,
            'read_lengths': read_lengths_list,
            'empirical_sub_rate': sub_rate,
            'empirical_indel_rate': indel_rate,
            'total_reads': len(substitutions_per_read),
        }

        print(f"\nProcessed {dirname} ({label}):")
        print(f"  Reads: {len(substitutions_per_read):,}")
        print(f"  Empirical sub rate: {sub_rate * 100:.4f}%")
        print(f"  Empirical indel rate: {indel_rate * 100:.6f}%")

    if not data_by_profile:
        print("No art_modern test set data found.")
        return

    all_subs = np.concatenate([d['substitutions'] for d in data_by_profile.values()])
    all_indels = np.concatenate([d['indels'] for d in data_by_profile.values()])
    max_subs = int(all_subs.max())
    max_indels = int(all_indels.max())
    bins_subs = np.arange(0, max_subs + 2) - 0.5

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle('Test sets simulated with art_modern', fontsize=11, y=0.98)

    # Plot 1: Substitution distribution
    ax1 = axes[0]
    for i, (label, data) in enumerate(data_by_profile.items()):
        empirical_pct = data['empirical_sub_rate'] * 100
        ax1.hist(data['substitutions'], bins=bins_subs, alpha=0.5, color=colors[i],
                 label=f"{label} (empirical = {empirical_pct:.2f}\\%)", edgecolor='none')

    ax1.set_xlabel('Number of Substitution Errors per Read', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_xlim(left=-0.5)

    def sci_formatter(x, pos):
        if x == 0:
            return '0'
        exp = int(np.floor(np.log10(abs(x))))
        coef = x / 10**exp
        if coef == 1:
            return f'$10^{{{exp}}}$'
        return f'${coef:.0f}\\times10^{{{exp}}}$'

    ax1.yaxis.set_major_formatter(FuncFormatter(sci_formatter))
    ax1.tick_params(labelsize=11)
    ax1.legend(title='Quality profile (substitution error rate)', fontsize=10, title_fontsize=10)

    # Plot 2: Indel distribution
    ax2 = axes[1]
    indel_values = np.arange(0, max_indels + 1)
    n_profiles = len(data_by_profile)
    bar_width = 0.8 / n_profiles

    for i, (label, data) in enumerate(data_by_profile.items()):
        empirical_pct = data['empirical_indel_rate'] * 100
        counts = [np.sum(data['indels'] == v) for v in indel_values]
        x_positions = indel_values + (i - n_profiles / 2 + 0.5) * bar_width
        ax2.bar(x_positions, counts, width=bar_width, color=colors[i],
                label=f"{label} (empirical = {empirical_pct:.4f}\\%)", alpha=0.8)

    ax2.set_yscale('log')
    ax2.set_xlabel('Number of Indel Errors per Read', fontsize=12)
    ax2.set_ylabel('Frequency (log scale)', fontsize=12)
    ax2.set_xticks(indel_values)
    ax2.tick_params(labelsize=11)
    ax2.legend(title='Quality profile (indel error rate)', fontsize=10, title_fontsize=10)

    plt.tight_layout()
    os.makedirs("../../illustrations/testset_error_rates_plots", exist_ok=True)
    output_filename = "../../illustrations/testset_error_rates_plots/error_distributions_art_modern.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_filename}")


if __name__ == "__main__":
    ### Get plots of sequence error distributions ###

    #Test sets generated with Mason
    results = analyze_all_csv_files()

    #Test sets generated with art_modern
    analyze_art_modern_testsets()
