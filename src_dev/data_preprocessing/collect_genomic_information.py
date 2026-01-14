import os
import shutil

import numpy as np
import pandas as pd
from Bio import SeqIO


def process_taxonomic_and_genomic_info() -> None:
    """
    Purpose: Merge taxonomical information with genomic statistical information for each organism in dataset and remove organisms with no family-level classification
    """
    os.makedirs("../../data/processed_data/dataset_information", exist_ok=True)
    genomes_data_dict = dict()

    with open("../../data/processed_data/taxonomy/genomes_tax_info.tab", "r") as infile_tax_data:
        # Iterate over taxonomic informaton per organism
        for line in infile_tax_data:
            info_attr = line.strip().split("|")
            accession = info_attr[0].strip("\t")

            # Extract taxonomical information: strain, species, genus, family, order, class, phylum, domain per organism
            # Initialzie inner dict for accession ID
            genomes_data_dict[accession] = dict()
            genomes_data_dict[accession]["strain"] = None
            genomes_data_dict[accession]["species"] = None
            genomes_data_dict[accession]["genus"] = None
            genomes_data_dict[accession]["family"] = None
            genomes_data_dict[accession]["order"] = None
            genomes_data_dict[accession]["class"] = None
            genomes_data_dict[accession]["phylum"] = None
            genomes_data_dict[accession]["domain"] = None

            # Skip accession number; loop over information on taxonomic ranks
            for tax_level_info_str in info_attr[1:]:
                if "\tstrain\t" in tax_level_info_str:
                    strain_name = tax_level_info_str.strip().split("\t")[-1]
                    genomes_data_dict[accession]["strain"] = strain_name
                elif "\tspecies\t" in tax_level_info_str:
                    species_name = tax_level_info_str.strip().split("\t")[-1]
                    genomes_data_dict[accession]["species"] = species_name
                elif "\tgenus\t" in tax_level_info_str:
                    genus_name = tax_level_info_str.strip().split("\t")[-1]
                    genomes_data_dict[accession]["genus"] = genus_name
                elif "\tfamily\t" in tax_level_info_str:
                    family_name = tax_level_info_str.strip().split("\t")[-1]
                    genomes_data_dict[accession]["family"] = family_name
                elif "\torder\t" in tax_level_info_str:
                    order_name = tax_level_info_str.strip().split("\t")[-1]
                    genomes_data_dict[accession]["order"] = order_name
                elif "\tclass\t" in tax_level_info_str:
                    class_name = tax_level_info_str.strip().split("\t")[-1]
                    genomes_data_dict[accession]["class"] = class_name
                elif "\tphylum\t" in tax_level_info_str:
                    phylum_name = tax_level_info_str.strip().split("\t")[-1]
                    genomes_data_dict[accession]["phylum"] = phylum_name
                elif "\tdomain\t" in tax_level_info_str:
                    domain_name = tax_level_info_str.strip().split("\t")[-1]
                    genomes_data_dict[accession]["domain"] = domain_name

            # Make sure "family" is present, otherwise delete data for that organism
            if genomes_data_dict[accession]["family"] is None:
                if os.path.exists(f"../../data/raw_data/genome_data/{accession}"):
                    print(f"Data with accession: {accession} removed due to lack of family-level classification.")
                    shutil.rmtree(f"../../data/raw_data/genome_data/{accession}")

                genbank_accession = accession.replace("GCF", "GCA")
                if os.path.exists(f"../../data/raw_data/genome_data/{genbank_accession}"):
                    print(f"Data with accession: {genbank_accession} removed due to lack of family-level classification.")
                    shutil.rmtree(f"../../data/raw_data/genome_data/{genbank_accession}")

            # If accession is classified on family level, calculate and add statistical metadata
            else:
                # Initialize
                genomes_data_dict[accession]["gc_content"] = None
                genomes_data_dict[accession]["genome_length_kb"] = None
                genomes_data_dict[accession]["coding_percentage"] = None
                genomes_data_dict[accession]["protein_coding_genes"] = None
                genomes_data_dict[accession]["cds_median_length"] = None
                genome_seq_len = 0
                at_count = 0
                gc_count = 0
                cds_positions_sum = 0
                cds_end = 0
                genes_count = 0
                cds_coords_overlap_dict = []
                landmarks = []
                cds_lengths = []

                # Extract filename and -path of file with genomic sequence(s) (assemblies: chromosome(s), plasmid(s))
                genome_files = os.listdir(f"../../data/raw_data/genome_data/{accession}/")
                genome_fasta_file = [file for file in genome_files if file.startswith(accession)][0]
                genome_filename = f"../../data/raw_data/genome_data/{accession}/{genome_fasta_file}"

                # Calculate GC content and collect genome size for each organism
                # Loop over each assembly sequence in genomic file
                for record in SeqIO.parse(genome_filename, "fasta-pearson"):
                    # Get GC-distribution
                    seq_upper = record.seq.upper()
                    at_count += seq_upper.count("A") + seq_upper.count("T")
                    gc_count += seq_upper.count("C") + seq_upper.count("G")

                    seq_len = len(record.seq)
                    genome_seq_len += seq_len

                genome_length_kb = int(genome_seq_len / 1000)
                gc_content = round(gc_count / (gc_count + at_count) * 100, 1)

                # Get fraction of genome positions which is coding (CDS annotations) and number of protein-coding genes (CDS annotations)
                with open(f"../../data/raw_data/genome_data/{accession}/genomic.gff", "r") as infile_genomic_gff:
                    for line in infile_genomic_gff:
                        # Skip comment lines
                        if len(line.split("\t")) < 7:
                            continue
                        else:
                            info_attributes = line.split("\t")
                            attr = info_attributes[2]
                            landmark = info_attributes[0]

                            if attr == "CDS":
                                # Count number of protein-coding genes (CDS annotations)
                                genes_count += 1

                                # Get CDS start and end coordinates
                                cds_start = int(info_attributes[3])
                                cds_end = int(info_attributes[4])

                                cds_length = cds_end - cds_start + 1
                                cds_lengths.append(cds_length)

                                if cds_coords_overlap_dict == []:
                                    cds_coords_overlap_dict.append([cds_start, cds_end])

                                # Extend CDS region if 2 CDSs overlap
                                elif cds_start <= cds_coords_overlap_dict[-1][-1] and landmark in landmarks:
                                    cds_coords_overlap_dict[-1][-1] = cds_end

                                # Add new CDS region if no overlap
                                else:
                                    cds_coords_overlap_dict.append([cds_start, cds_end])

                                # Keep track of landmark (chromosome or plasmid) to which CDS belongs
                                if landmark not in landmarks:
                                    landmarks.append(landmark)

                # Add count for number of positions being CDS and calculate overall coding percentage of genome
                for cds_start, cds_end in cds_coords_overlap_dict:
                    cds_positions_sum += cds_end - cds_start + 1

                coding_percentage = round((cds_positions_sum / genome_seq_len) * 100, 2)

                assert coding_percentage != 0.0, f"Coding percentage for organism with accession {accession} is 0.0% - please check GFF file."
                assert genes_count != 0, f"Number of protein-coding genes for organism with accession {accession} is 0 - please check GFF file."
                assert genome_length_kb != 0, f"Genome length for organism with accession {accession} is 0 kb - please check FASTA file."

                cds_median_length = int(np.median(cds_lengths))

                genomes_data_dict[accession]["gc_content"] = gc_content
                genomes_data_dict[accession]["genome_length_kb"] = genome_length_kb
                genomes_data_dict[accession]["coding_percentage"] = coding_percentage
                genomes_data_dict[accession]["protein_coding_genes"] = genes_count
                genomes_data_dict[accession]["cds_median_length"] = cds_median_length

    genomes_data_df = pd.DataFrame(genomes_data_dict).T

    # Remove metadata for entries with no family-level classification
    genomes_data_df = genomes_data_df[genomes_data_df["family"].notna()].copy()
    genomes_data_df.to_csv("../../data/processed_data/dataset_information/genomes_info.csv", sep=",", encoding="utf-8")


if __name__ == "__main__":
    process_taxonomic_and_genomic_info()
