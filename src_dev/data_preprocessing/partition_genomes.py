import os
from collections import defaultdict

import numpy as np
import pandas as pd

# Purpose: Make lists of genome IDs that go into train, val and test partitions


def partition_dataset(tax_info_genomes_df, testset_accessions, random_seed=42) -> dict:
    """
    Partition organisms into train, validation, and test sets.

    Args:
        tax_info_genomes_df (pd.DataFrame): DataFrame with organism information including 'accession', 'family',
                                        'gc_content', and 'domain' columns
        testset_accessions (set): Set of accessions to include in the test set
        random_seed (int): Random seed for reproducibility

    Returns:
        dict with keys 'train', 'val', 'test', 'excluded' containing lists of accessions
    """

    def __calculate_family_stats__(domain_df, family_groups) -> pd.DataFrame:
        family_stats = []
        for family, indices in family_groups.items():
            family_data = domain_df.loc[indices]
            family_stats.append({"family": family, "indices": indices, "count": len(indices), "mean_gc": family_data["gc_content"].mean()})

        return pd.DataFrame(family_stats).sort_values("mean_gc").reset_index(drop=True)

    def __select_validation_families__(family_stats_df, family_groups, domain_name) -> tuple:
        """Select families for validation set from one domain."""
        total_organisms = family_stats_df["count"].sum()
        target_val = total_organisms * 0.1  # Aim for ~10% of organisms remaining in validation set (after defining test set organisms)

        print(f"{domain_name}:")
        print(f"Total: {total_organisms} organisms, targeting ~{int(target_val)} for validation")

        n_families = len(family_stats_df)

        if n_families == 0:
            return [], []

        val_families = []
        val_indices = []
        val_count = 0
        selected_family_indices = set()

        # Calculate number of families to sample (start with ~12% of families, some are larger than others)
        n_to_sample = max(1, int(n_families * 0.12))

        # First pass: evenly sample across GC range
        for i in range(n_to_sample):
            fam_idx = int(i * n_families / n_to_sample)
            if fam_idx >= n_families:
                continue

            family_info = family_stats_df.iloc[fam_idx]
            val_families.append(family_info["family"])
            val_indices.extend(family_groups[family_info["family"]])
            val_count += family_info["count"]
            selected_family_indices.add(fam_idx)

        # Second pass: add more families if needed to reach ~10%
        # Add families from gaps in GC distribution to maintain balance
        if val_count < target_val * 0.95:
            # Find unselected families and sort by how well they fill gaps
            unselected = []
            for fam_idx in range(n_families):
                if fam_idx not in selected_family_indices:
                    unselected.append(fam_idx)

            # For each unselected family, calculate its distance to nearest selected family
            # Prioritize families that are far from selected ones (fill gaps)
            def __gap_score__(idx):
                if not selected_family_indices:
                    return 0
                min_dist = min(abs(idx - sel) for sel in selected_family_indices)
                return min_dist

            unselected.sort(key=__gap_score__, reverse=True)

            for fam_idx in unselected:
                if val_count >= target_val * 0.95:
                    break

                family_info = family_stats_df.iloc[fam_idx]

                # Don't add if it overshoots too much
                if val_count + family_info["count"] > target_val * 1.1:
                    continue

                val_families.append(family_info["family"])
                val_indices.extend(family_groups[family_info["family"]])
                val_count += family_info["count"]
                selected_family_indices.add(fam_idx)

        print(f"Validation: {len(val_families)} families, {val_count} organisms ({val_count / total_organisms * 100:.1f}%)")

        return val_families, val_indices

    np.random.seed(random_seed)

    # Ensure that all organisms have a family classification
    missing_family_mask = tax_info_genomes_df["family"].isna()
    missing_family_df = tax_info_genomes_df[missing_family_mask].copy()

    assert len(missing_family_df) == 0, "There are organisms with missing family information."

    # Step 1: Identifyall families with tesset accessions
    test_organisms = tax_info_genomes_df[tax_info_genomes_df["accession"].isin(testset_accessions)]
    test_families = set(test_organisms["family"].dropna().unique())

    print("Step 1 complete: Extracted test set families.")
    print(f"Test set families ({len(test_families)}): {test_families}")
    print()

    # Step 2: Assign all organisms from test families to test set
    test_mask = tax_info_genomes_df["family"].isin(test_families)
    test_df = tax_info_genomes_df[test_mask].copy()
    remaining_df = tax_info_genomes_df[~test_mask].copy()

    print("Step 2 complete: Assigned all organisms from test families to test set.")
    print(f"Test set: {len(test_df)} organisms")
    print(f"Remaining: {len(remaining_df)} organisms")
    print()

    # Step 3: Separate organisms by domain
    archaea_df = remaining_df[remaining_df["domain"] == "Archaea"].copy()
    bacteria_df = remaining_df[remaining_df["domain"] == "Bacteria"].copy()

    print("Step 3 complete: Separated remaining organisms by domain.")
    print(f"Archaea: {len(archaea_df)} organisms")
    print(f"Bacteria: {len(bacteria_df)} organisms")
    print()

    # Step 4: Group by family within each domain
    def group_by_family(domain_df) -> dict:
        family_groups = defaultdict(list)
        for idx, row in domain_df.iterrows():
            family = row["family"]
            family_groups[family].append(idx)
        return family_groups

    archaea_family_groups = group_by_family(archaea_df)
    bacteria_family_groups = group_by_family(bacteria_df)

    print("Step 4 complete: Grouped organisms by family within each domain.")
    print(f"Archaeal families: {len(archaea_family_groups)}")
    print(f"Bacterial families: {len(bacteria_family_groups)}")
    print()

    # Step 5: Calculate statistics for each family within each domain, respectively
    archaea_family_stats = __calculate_family_stats__(archaea_df, archaea_family_groups)
    bacteria_family_stats = __calculate_family_stats__(bacteria_df, bacteria_family_groups)

    print("Step 5 complete: Calculated family statistics for each domain.")
    print()

    # Step 6: Select validation families for each domain independently
    # Select validation sets for each domain
    print("Step 6 complete: Selected validation families for each domain.")
    archaea_val_families, archaea_val_indices = __select_validation_families__(archaea_family_stats, archaea_family_groups, "Archaea")
    bacteria_val_families, bacteria_val_indices = __select_validation_families__(bacteria_family_stats, bacteria_family_groups, "Bacteria")
    print()

    # Step 7: Create training sets (all non-validation families)
    archaea_train_families = [f for f in archaea_family_groups.keys() if f not in archaea_val_families]
    bacteria_train_families = [f for f in bacteria_family_groups.keys() if f not in bacteria_val_families]

    archaea_train_indices = []
    for family in archaea_train_families:
        archaea_train_indices.extend(archaea_family_groups[family])

    bacteria_train_indices = []
    for family in bacteria_train_families:
        bacteria_train_indices.extend(bacteria_family_groups[family])

    print("Step 7 complete: Created training sets for each domain.")
    print()

    # Step 8: Create final dataframes for each partition
    train_df = pd.concat([archaea_df.loc[archaea_train_indices], bacteria_df.loc[bacteria_train_indices]])

    val_df = pd.concat([archaea_df.loc[archaea_val_indices], bacteria_df.loc[bacteria_val_indices]])

    # Combine family lists for reference
    train_families = archaea_train_families + bacteria_train_families
    val_families = archaea_val_families + bacteria_val_families

    # Print statistics for each data partition to assess balance
    print("\n" + "=" * 60)
    print("PARTITION STATISTICS")
    print("=" * 60)

    print(f"\nTest Set:")
    print(f"  Total organisms: {len(test_df)}")
    print(f"  Families: {len(test_families)}")
    print(f"  Archaea: {(test_df['domain'] == 'Archaea').sum()}")
    print(f"  Bacteria: {(test_df['domain'] == 'Bacteria').sum()}")
    print(f"  GC content range: {test_df['gc_content'].min():.1f} - {test_df['gc_content'].max():.1f}")
    print(f"  GC content mean: {test_df['gc_content'].mean():.1f}")
    print(f"  GC content median: {test_df['gc_content'].median():.1f}")

    print(f"\nTraining Set:")
    print(f"  Total organisms: {len(train_df)}")
    print(f"  Families: {len(train_families)}")
    train_archaea = (train_df["domain"] == "Archaea").sum()
    train_bacteria = (train_df["domain"] == "Bacteria").sum()
    print(f"  Archaea: {train_archaea} ({train_archaea / len(archaea_df) * 100:.1f}% of archaea)")
    print(f"  Bacteria: {train_bacteria} ({train_bacteria / len(bacteria_df) * 100:.1f}% of bacteria)")
    print(f"  GC content range: {train_df['gc_content'].min():.1f} - {train_df['gc_content'].max():.1f}")
    print(f"  GC content mean: {train_df['gc_content'].mean():.1f}")
    print(f"  GC content median: {train_df['gc_content'].median():.1f}")

    print(f"\nValidation Set:")
    print(f"  Total organisms: {len(val_df)}")
    print(f"  Families: {len(val_families)}")
    val_archaea = (val_df["domain"] == "Archaea").sum()
    val_bacteria = (val_df["domain"] == "Bacteria").sum()
    print(f"  Archaea: {val_archaea} ({val_archaea / len(archaea_df) * 100:.1f}% of archaea)")
    print(f"  Bacteria: {val_bacteria} ({val_bacteria / len(bacteria_df) * 100:.1f}% of bacteria)")
    print(f"  GC content range: {val_df['gc_content'].min():.1f} - {val_df['gc_content'].max():.1f}")
    print(f"  GC content mean: {val_df['gc_content'].mean():.1f}")
    print(f"  GC content median: {val_df['gc_content'].median():.1f}")
    print("\n" + "=" * 60 + "\n")

    # Return accessions for each partition
    return {"train": train_df["accession"].tolist(), "val": val_df["accession"].tolist(), "test": test_df["accession"].tolist(), "train_df": train_df, "val_df": val_df, "test_df": test_df}


def main() -> None:
    """
    Main function to partition genomes and save accession lists to files.
    1. Load genome taxonomic information from CSV.
    2. Define test set accessions.
    3. Partition dataset into train, val, and test sets.
    4. Save accession lists to text files.
    """

    tax_info_genomes_df = pd.read_csv("../../data/processed_data/dataset_information/genomes_info.csv", index_col=None).rename(columns={"Unnamed: 0": "accession"})

    # Define the accessions we want to include in the test set from both archaeal and bacterial species
    testset_accessions_archaea = {
        "GCF_000011125.1",  # 56.3
        "GCF_000008665.1",  # 48.6
        "GCF_004799605.1",  # 66.3
        "GCF_000017165.1",  # 31.3
        "GCF_000007345.1",  # 42.7
        "GCF_000012545.1",  # 27.6
    }

    testset_accessions_bacteria = {
        "GCF_000007365.1",  # 25.3
        "GCF_000009045.1",  # 43.5
        "GCF_025998455.1",  # 38.5
        "GCF_000195955.2",  # 66.5
        "GCF_020736045.1",  # 38.2
        "GCF_000012765.1",  # 23.8, translation table 4
        "GCF_028609885.1",  # 61.4
        "GCF_000005845.2",  # 50.8
        "GCF_000006765.1",  # 66.6
        "GCF_000013425.1",  # 32.9
    }

    testset_accessions = testset_accessions_archaea | testset_accessions_bacteria

    # Run function, get partitions
    partitions = partition_dataset(tax_info_genomes_df, testset_accessions)

    # Save to files
    os.makedirs("../../data/processed_data/genome_partitions/", exist_ok=True)
    with open("../../data/processed_data/genome_partitions/train_partition_accessions.txt", "w") as f:
        f.write("\n".join(partitions["train"]))

    with open("../../data/processed_data/genome_partitions/val_partition_accessions.txt", "w") as f:
        f.write("\n".join(partitions["val"]))

    with open("../../data/processed_data/genome_partitions/test_partition_accessions.txt", "w") as f:
        f.write("\n".join(partitions["test"]))


if __name__ == "__main__":
    main()
