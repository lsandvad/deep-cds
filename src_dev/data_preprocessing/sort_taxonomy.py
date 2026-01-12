import os
import zipfile
import io
from tqdm import tqdm

os.makedirs("../../data/processed_data/taxonomy/", exist_ok=True)

def extract_species_tax_ids():
	"""
	Extracts species taxonomic IDs from GFF files in the genome data directory.
	
	Returns:
        dict: A dictionary mapping accession numbers to their corresponding species taxonomic IDs.
	"""
	
	species_tax_ids = dict()
	
	accessions = os.listdir("../../data/raw_data/genome_data/")
	#Only extract accessions with RefSeq annotations (starts with GCF_)
	accessions = [accession for accession in accessions if accession.startswith("GCF_")]

    #Iterate through each accession and extract the species taxonomic ID from the GFF file
	for accession in accessions:
		gff_file = open(f"../../data/raw_data/genome_data/{accession}/genomic.gff", "r")
		for line in gff_file:
			if line.startswith("##species"):
				species_tax_id = line.strip().split("=")[-1]
				#Store the species taxonomic ID in the dictionary with the accession as key
				species_tax_ids[accession] = species_tax_id
				break
		gff_file.close()

	return species_tax_ids


def sort_organisms_taxonomies(species_tax_ids):
	"""
	Create a taxonomical representation for each organism based on several levels.

	Args:
		species_tax_ids (dict): A dictionary mapping accession numbers to their corresponding species taxonomic IDs.
	
	Output:
		A tab-separated file with taxonomical information about each organism
	"""

	#Open file with node-data, extract the following:
		#node (organism ID)
		#parent node (organism ID)
		#taxonomic rank

	#Requires: 
    #taxdmp.zip downloaded from: https://ftp.ncbi.nih.gov/pub/taxonomy/

	print("Extracting taxonomical information from NCBI Taxonomy database...")

	with zipfile.ZipFile('../../data/raw_data/taxonomy_data/taxdmp.zip', 'r') as zip_file:
		with zip_file.open('taxdmp/nodes.dmp', 'r') as file:
			nodes = io.TextIOWrapper(file, encoding='utf-8')

			#Initialize
			nodes_list = []
			ranks_list = []

			#Extract species nodes (Organism ID)
			for line in nodes:
				node_line = line.split("\t")
				
				#Create list of lists, first element in each inner list is node,
				#second element in parent node, third element is rank
				nodes_list.append([node_line[0],node_line[2],node_line[4]])
				
				#Save all ranks occuring in a list
				ranks_list.append(node_line[4])

			nodes.close()

		with zip_file.open('taxdmp/names.dmp', 'r') as file:
			#Open file with taxonomy names-data. For the "scientific name" lines extract:
				#node (organism/rank ID)
				#node name (Scientific name)
			names = io.TextIOWrapper(file, encoding='utf-8')

			#Initialize
			names_list = []

			for line in names:
				name_line = line.split("\t")
				if name_line[6] == "scientific name":
					names_list.append([name_line[0],name_line[2]])

			names.close()


	#Initialize
	output_taxa = open("../../data/processed_data/taxonomy/genomes_tax_info.tab", "w")
	count_org = 0
	ranks_identified = dict()
	
    #Iterate over each genome (accession) and extract taxonomical information
	for accession, species_tax_id in tqdm(species_tax_ids.items(), desc="Processing organisms", unit="organisms"):
		#Initialize
		Searching = True
		count_org += 1
		counter = 0

		#Search for taxonomical information
		while Searching: 
			try: 
				#If a nodes' ID corresponds to the organism, write organism to file
				if nodes_list[counter][0] == species_tax_id:
					#Write organism, proteome ID and protein count to file
					output_taxa.write(accession + "\t" + "|" + "\t")

					#Initialize
					counter_names = 0
					Searching_names = True

					while Searching_names:
						#Write rank and genkbank ID of rank and rank name to file
						if nodes_list[counter][0] == names_list[counter_names][0]:
							output_taxa.write(nodes_list[counter][2] + "\t" + nodes_list[counter][0] + "\t")
							output_taxa.write(names_list[counter_names][1] + "\t" + "|")

							if nodes_list[counter][2] not in ranks_identified:
								ranks_identified[nodes_list[counter][2]] = [[nodes_list[counter][0], names_list[counter_names][1]]]
							elif nodes_list[counter][2] in ranks_identified:
								if nodes_list[counter][0] not in ranks_identified[nodes_list[counter][2]]:
									ranks_identified[nodes_list[counter][2]].append([nodes_list[counter][0], names_list[counter_names][1]])
							

							Searching_names = False
						
						counter_names += 1

					#Save parent node to the particular node
					parent_node = nodes_list[counter][1]

					#Initialize
					counter = 0
					counter_names = 0
					Searching_names = True

					#Search for parent node; add this, and taxonomic rank name and number to file
					while Searching:
						if nodes_list[counter][0] == parent_node:
							Searching_names = True
							while Searching_names:
								if nodes_list[counter][0] == names_list[counter_names][0]:
									output_taxa.write("\t" + nodes_list[counter][2] + "\t" + nodes_list[counter][0])
									output_taxa.write("\t" + names_list[counter_names][1] + "\t" + "|")

									if nodes_list[counter][2] not in ranks_identified:
										ranks_identified[nodes_list[counter][2]] = [[nodes_list[counter][0], names_list[counter_names][1]]]
									elif nodes_list[counter][2] in ranks_identified:
										if nodes_list[counter][0] not in ranks_identified[nodes_list[counter][2]]:
											ranks_identified[nodes_list[counter][2]].append([nodes_list[counter][0], names_list[counter_names][1]])
							
									#Break loops when superkingdom (highest rank) is reached
									if int(nodes_list[counter][0]) == 2 or int(nodes_list[counter][0]) == 2157:  #2 is domain bacteria, 2157 is domain archaea
										#Start newline in file for next organism
										output_taxa.write("\n")
										Searching = False
									Searching_names = False
								
								counter_names += 1
							parent_node = nodes_list[counter][1]

							#Initialize
							counter = 0
							counter_names = 0

						counter += 1
					Searching = False
				counter += 1

			except IndexError:
				print("Issue extracting taxonomical information for accession:")
				print(accession)
				break 


	#Close files
	output_taxa.close()

	print(f"Taxonomical information extracted for {count_org} organisms.")

if __name__ == "__main__":
	#Extract species taxonomic IDs from GFF files
	species_tax_ids = extract_species_tax_ids()
	sort_organisms_taxonomies(species_tax_ids)