import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def build_taxonomy_tree(df, rank_columns):
    """
    Build a hierarchical tree structure from taxonomic ranks.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with taxonomy data
    rank_columns : list
        List of column names in taxonomic order (e.g., ['phylum', 'class', 'order', 'family', 'genus', 'species'])
    
    Returns:
    --------
    dict : Nested dictionary representing the tree structure
    """

    #Initialize
    tree = {}
    
    # Iterate through each row in the DataFrame to build the tree using taxonomic ranks
    for idx, row in df.iterrows():
        current_level = tree
        
        for rank in rank_columns:
            taxon = row[rank]
            
            # Handle missing values
            if pd.isna(taxon) or taxon == '':
                taxon = f"Unknown_{rank}"
            
            if taxon not in current_level:
                current_level[taxon] = {}
            
            current_level = current_level[taxon]
        
        # Add leaf node with accession
        accession = row['accession']
        current_level[accession] = None
    
    return tree


def sanitize_name(name):
    """
    Sanitize names for Newick format by replacing problematic characters.
    """
    # Replace or remove problematic characters
    replacements = {
        '(': '[',
        ')': ']',
        ',': '_',
        ':': '_',
        ';': '_',
        ' ': '_',
        "'": '',
        '"': ''
    }
    
    for old, new in replacements.items():
        name = str(name).replace(old, new)
    
    return name

def tree_to_newick(tree_dict, name="root"):
    """
    Convert nested dictionary tree to Newick format string.
    
    Parameters:
    -----------
    tree_dict : dict
        Nested dictionary from build_taxonomy_tree
    name : str
        Name of current node
    
    Returns:
    --------
    str : Newick format string
    """
    # Sanitize the node name
    safe_name = sanitize_name(name)
    
    if tree_dict is None:
        # Leaf node
        return safe_name
    
    if not tree_dict:
        # Empty node
        return safe_name
    
    # Internal node with children
    children = [tree_to_newick(subtree, child_name) 
                for child_name, subtree in tree_dict.items()]
    
    return f"({','.join(children)}){safe_name}"


def create_figtree_annotations(df, accession_col='accession', partition_col='partition'):
    """
    Create annotation file for FigTree to color branches by partition.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with accession and partition columns
    accession_col : str
        Name of accession column
    partition_col : str
        Name of partition column (train/val/test)
    
    Returns:
    --------
    pandas DataFrame : Annotation table for FigTree
    """
    annotations = df[[accession_col, partition_col]].copy()
    annotations.columns = ['taxa', 'partition']
    return annotations


# Main execution
if __name__ == "__main__":
    
    # Load data
    df_taxonomy = pd.read_csv("../../data/processed_data/dataset_information/genomes_info.csv")
    df_taxonomy = df_taxonomy.rename(columns={"Unnamed: 0":"accession"})

    # Load partition accessions
    train_accessions = open("../../data/processed_data/genome_partitions/train_partition_accessions.txt").read().splitlines()
    val_accessions = open("../../data/processed_data/genome_partitions/val_partition_accessions.txt").read().splitlines()
    test_accessions = open("../../data/processed_data/genome_partitions/test_partition_accessions.txt").read().splitlines()

    #Initialize
    os.makedirs("../../data/processed_data/dataset_information/taxonomy_trees_with_colored_partitions/", exist_ok=True)
    df_taxonomy['partition'] = 'none'

    # Assign partitions and save updated taxonomy information with partitions
    df_taxonomy.loc[df_taxonomy['accession'].isin(train_accessions), 'partition'] = 'train'
    df_taxonomy.loc[df_taxonomy['accession'].isin(val_accessions), 'partition'] = 'val'
    df_taxonomy.loc[df_taxonomy['accession'].isin(test_accessions), 'partition'] = 'test'
    df_taxonomy.to_csv("../../data/processed_data/dataset_information/genomes_info_with_partitions.csv", index=False)

    df_taxonomy_archaea = df_taxonomy[df_taxonomy['domain'] == 'Archaea'].reset_index(drop=True)
    df_taxonomy_bacteria = df_taxonomy[df_taxonomy['domain'] == 'Bacteria'].reset_index(drop=True)

    # Define taxonomic rank columns in order (adjust to match data)
    rank_columns = ['phylum', 'class', 'order', 'family', 'genus', 'species']

    # Build the tree
    tree = build_taxonomy_tree(df_taxonomy_archaea, rank_columns)

    # Convert to Newick format
    newick_string = tree_to_newick(tree) + ";"

    # Save Newick tree
    with open('../../data/processed_data/dataset_information/taxonomy_trees_with_colored_partitions/taxonomy_tree_archaea.nwk', 'w') as f:
        f.write(newick_string)

    # Create annotation file for FigTree
    annotations = create_figtree_annotations(df_taxonomy_archaea)
    annotations.to_csv('../../data/processed_data/dataset_information/taxonomy_trees_with_colored_partitions/figtree_annotations_archaea.txt', sep='\t', index=False)

    print("Annotations saved to 'figtree_annotations_archaea.txt'")
    print(f"\nTree contains {len(df_taxonomy_archaea)} taxa")
    print(f"Partition distribution:\n{df_taxonomy_archaea['partition'].value_counts()}")


    # Build the tree
    tree = build_taxonomy_tree(df_taxonomy_bacteria, rank_columns)

    # Convert to Newick format
    newick_string = tree_to_newick(tree) + ";"

    # Save Newick tree
    with open('../../data/processed_data/dataset_information/taxonomy_trees_with_colored_partitions/taxonomy_tree_bacteria.nwk', 'w') as f:
        f.write(newick_string)

    # Create annotation file for FigTree
    annotations = create_figtree_annotations(df_taxonomy_bacteria)
    annotations.to_csv('../../data/processed_data/dataset_information/taxonomy_trees_with_colored_partitions/figtree_annotations_bacteria.txt', sep='\t', index=False)

    print("Annotations saved to 'figtree_annotations_bacteria.txt'")
    print(f"\nTree contains {len(df_taxonomy_bacteria)} taxa")
    print(f"Partition distribution:\n{df_taxonomy_bacteria['partition'].value_counts()}")