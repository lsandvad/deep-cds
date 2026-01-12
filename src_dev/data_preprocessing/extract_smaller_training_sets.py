import pandas as pd
import numpy as np

def create_stratified_subsamples(genomes_info_train, target_sizes=[400, 200, 100]):
    """
    Create stratified subsamples maintaining family distribution proportionally.
    Always includes genomes with extreme (min and max) GC content.

    Args:
        df (pd.DataFrame): DataFrame with columns 'family', 'gc_content', 'domain'.
        target_sizes (list): List of target dataset sizes.

    Returns: 
        dict: Dictionary with keys as target sizes and values as sampled DataFrames.
    """

    np.random.seed(1)  #For reproducibility
    results = {}
    
    #Identify the most extreme GC content genomes (must be kept in all subsets)
    min_gc_idx = genomes_info_train['gc_content'].idxmin()
    max_gc_idx = genomes_info_train['gc_content'].idxmax()
    extreme_gc_indices = [min_gc_idx, max_gc_idx]
    
    #Calculate family counts and proportions
    family_counts = genomes_info_train['family'].value_counts()
    family_proportions = family_counts / len(genomes_info_train)
    
    #Iterate over each target size
    for target_size in sorted(target_sizes, reverse=True):
        print(f"Creating {target_size} genome dataset with proportional sampling...")
        
        #Start with min,max GC genomes
        sampled_indices = extreme_gc_indices.copy()
        extreme_families = set(genomes_info_train.loc[extreme_gc_indices, 'family'].values)
        
        #Calculate target samples per family (proportional to original distribution)
        family_targets = {}
        for family, proportion in family_proportions.items():
            target_count = proportion * target_size
            #Round but ensure at least 1 if family is represented by extreme GC
            if family in extreme_families:
                family_targets[family] = max(1, int(np.round(target_count)))
            else:
                family_targets[family] = int(np.round(target_count))
        
        #Adjust targets to sum to target_size (accounting for extreme GC already selected)
        total_targeted = sum(family_targets.values())
        if total_targeted != target_size:
            #Distribute difference proportionally
            diff = target_size - total_targeted
            #Sort families by their decimal remainder to decide which gets adjusted
            remainders = [(fam, (family_proportions[fam] * target_size) % 1) 
                         for fam in family_targets.keys() if family_targets[fam] > 0]
            remainders.sort(key=lambda x: x[1], reverse=(diff > 0))
            
            for i in range(abs(diff)):
                fam = remainders[i % len(remainders)][0]
                if diff > 0:
                    family_targets[fam] += 1
                elif family_targets[fam] > 1 or fam not in extreme_families:
                    family_targets[fam] = max(0, family_targets[fam] - 1)
        
        #Sample from each family according to targets
        for family, target_count in family_targets.items():
            if target_count == 0:
                continue
                
            family_df = genomes_info_train[genomes_info_train['family'] == family]
            
            #Check how many already sampled from this family (via extreme GC)
            current_from_family = sum(1 for idx in sampled_indices if genomes_info_train.loc[idx, 'family'] == family)
            additional_needed = target_count - current_from_family
            
            if additional_needed > 0:
                #Get available genomes from this family
                available = family_df.index.difference(sampled_indices)
                
                if len(available) > 0:
                    n_to_sample = min(additional_needed, len(available))
                    samples = np.random.choice(available, n_to_sample, replace=False)
                    sampled_indices.extend(samples.tolist())
        
        #Final adjustment to ensure extreme GC genomes are included
        sampled_indices = list(set(sampled_indices))
        for idx in extreme_gc_indices:
            if idx not in sampled_indices:
                sampled_indices.append(idx)
        
        #If we're still not at target size, fill remaining slots proportionally
        if len(sampled_indices) < target_size:
            remaining = target_size - len(sampled_indices)
            available = genomes_info_train.index.difference(sampled_indices)
            if len(available) > 0:
                #Sample remaining from available genomes, weighted by family proportion
                available_df = genomes_info_train.loc[available]
                weights = available_df['family'].map(family_proportions).values
                weights = weights / weights.sum()  # Normalize
                
                additional = np.random.choice(
                    available, 
                    min(remaining, len(available)), 
                    replace=False,
                    p=weights
                )
                sampled_indices.extend(additional.tolist())
        
        #If we exceeded (shouldn't happen, but just in case), trim excess
        if len(sampled_indices) > target_size:
            # Keep extreme GC genomes, remove others
            excess = len(sampled_indices) - target_size
            removable = [idx for idx in sampled_indices if idx not in extreme_gc_indices]
            to_remove = np.random.choice(removable, excess, replace=False)
            sampled_indices = [idx for idx in sampled_indices if idx not in to_remove]
        
        results[target_size] = genomes_info_train.loc[sampled_indices].copy()
    
    return results


def print_dataset_summary(original_df, subsampled_dict):
    """
    Print summary statistics for each subsampled dataset.
    """
    print("="*60)
    print(f"ORIGINAL DATASET: {len(original_df)} genomes")
    print(f"  Families: {original_df['family'].nunique()}")
    
    # Family distribution by domain
    bacteria_families = original_df[original_df['domain'] == 'Bacteria']['family'].nunique()
    archaea_families = original_df[original_df['domain'] == 'Archaea']['family'].nunique()
    print(f"    Bacteria families: {bacteria_families}")
    print(f"    Archaea families: {archaea_families}")
    
    print(f"  Avg genomes per family: {len(original_df) / original_df['family'].nunique():.2f}")
    
    # GC content statistics
    print(f"  GC content range: {original_df['gc_content'].min():.1f} - {original_df['gc_content'].max():.1f}%")
    print(f"  GC content mean: {original_df['gc_content'].mean():.1f}%")
    print(f"  GC content median: {original_df['gc_content'].median():.1f}%")
    
    print(f"  Domains: {original_df['domain'].value_counts().to_dict()}")
    
    # Show family size distribution
    family_sizes = original_df['family'].value_counts()
    print(f"  Family size distribution:")
    print(f"    Min: {family_sizes.min()}, Max: {family_sizes.max()}, Median: {family_sizes.median():.0f}")
    print()
    
    for size in sorted(subsampled_dict.keys(), reverse=True):
        df = subsampled_dict[size]
        pct = (size / len(original_df)) * 100
        
        print(f"SUBSAMPLED DATASET: {size} genomes ({pct:.1f}% of original)")
        print(f"  Families: {df['family'].nunique()}")
        
        # Family distribution by domain
        bacteria_families = df[df['domain'] == 'Bacteria']['family'].nunique()
        archaea_families = df[df['domain'] == 'Archaea']['family'].nunique()
        print(f"    Bacteria families: {bacteria_families}")
        print(f"    Archaea families: {archaea_families}")
        
        print(f"  Avg genomes per family: {len(df) / df['family'].nunique():.2f}")
        
        # GC content statistics
        print(f"  GC content range: {df['gc_content'].min():.1f} - {df['gc_content'].max():.1f}%")
        print(f"  GC content mean: {df['gc_content'].mean():.1f}%")
        print(f"  GC content median: {df['gc_content'].median():.1f}%")
        
        print(f"  Domains: {df['domain'].value_counts().to_dict()}")
        
        # Show family size distribution
        family_sizes = df['family'].value_counts()
        print(f"  Family size distribution:")
        print(f"    Min: {family_sizes.min()}, Max: {family_sizes.max()}, Median: {family_sizes.median():.0f}")
        
        # Check proportionality
        original_domain_pcts = original_df['domain'].value_counts(normalize=True) * 100
        subsample_domain_pcts = df['domain'].value_counts(normalize=True) * 100
        print(f"  Domain proportions (Original → Subsample):")
        for domain in original_domain_pcts.index:
            orig_pct = original_domain_pcts[domain]
            sub_pct = subsample_domain_pcts.get(domain, 0)
            print(f"    {domain}: {orig_pct:.1f}% → {sub_pct:.1f}%")
        print()



accessions_train = open("../../data/processed_data/genome_partitions/train_partition_accessions.txt").read().splitlines()
genomes_info = pd.read_csv("../../data/processed_data/dataset_information/genomes_info.csv", index_col=None).rename(columns = {'Unnamed: 0': 'accession'})
genomes_info_train = genomes_info[genomes_info["accession"].isin(accessions_train)]

results = create_stratified_subsamples(genomes_info_train)
print_dataset_summary(genomes_info_train, results)

with open('../../data/processed_data/genome_partitions/train_partition_accessions_400_genomes.txt', 'w') as f:
    f.write('\n'.join(list(results[400]['accession'])))

with open('../../data/processed_data/genome_partitions/train_partition_accessions_200_genomes.txt', 'w') as f:
    f.write('\n'.join(list(results[200]['accession'])))

with open('../../data/processed_data/genome_partitions/train_partition_accessions_100_genomes.txt', 'w') as f:
    f.write('\n'.join(list(results[100]['accession'])))