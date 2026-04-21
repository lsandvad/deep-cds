# DeepCDS
Project workspace for DeepCDS project

# Installation

# Usage Instructions
DeepCDS can be run via the command line by cloning this repository and installing the required packages. 
To test the installation, run the following command from project root:
```
python3 ./predict_with_deepcds.py -model MODEL_FIX-in ./data_example/FILE_FIX.fasta
```

DeepCDS can be run to predict on your own data using the general command:
```
python3 ./predict_with_deepcds.py [optional arguments] -model MODEL -in INPUT_FILENAME 
```
Please note that the program uses the information in the /src, /models, and /configs directories. 

The input file (in FASTA format) and model type arguments are required. Additionally, DeepCDS accepts a range of optional arguments:

| Input Argument                      | Description                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-in`, `--input_filename`       | IMPLEMENT ME! Input file in FASTA format. The allowed input alphabet is A, C, G, T, U and N (unknown). All the other letters will be converted to N before processing. T and U are treated as equivalent. The input file can also be provided in gzipped version with a .gz extension.                                                                        |
|`--compute_device` | Which hardware accelerator to use. Options are:  `cuda` (NVIDIA GPU), `mps` (Apple Silicon), or `cpu`. The program will automatically fall back to CPU if the requested device is unavailable.|
| `--batch_size`    | Specifies the number of samples to process together in a single pass during prediction. Default value: `256`.                                     |
|`--min_cds_length` | Minimum length (nt) for predicted CDS sequences. We recommend not going below 30 nt as predictive performance below this threshold has not been evaluated. Default value: `60`|

- Giv eksempel
- Skriv hvad input er
- Beskriv input argumenter
- Skriv hvad output er! Forventer at vi outputter: GFF fil, 2x fasta filer med CDS på DNA-niveau og translaterede sekvenser

### TO DO
- [ ] Skriv et "Supplementary Note" afsnit i overleaf om outputs (gff og fastaformater)
- [ ] DOI på alle referencer
- [ ] Datasæt størrelser:
    - [ ] Skriv: "The final dataset consists of X sequences (x in train, y in val, z in test), with X positions labelled as ...?"
    - [ ] Read counts pr. genom i excel-fil som supplementary med train/val/test labeltag

### Manuskript; mangler
- [ ] Abstract
- [ ] Beskrivelse af datasæt størrelse og fordeling på sekvenstyper (se TO DO)
- Resultater: 
    - [ ] "Findings from abalations..." ESM-2 ablations (træner stadig)
        - [ ] 100, 200, 400 genomer?
        - [ ] Træningsdata størrelse: Skriv at vi har undersøgt performance som funktion af træningsdata størrelsen (antal genomer) og det ikke giver så meget fra XX til XX… (enkelt sætning)
- [ ] Diskussion

### Implementering af script til prediction
- [ ] Output fasta filer
    - [x] Nukleotidsekvens
    - [ ] Aminosyresekvens (NNN encodes som X; stop codon encodes som *)
- [x] Sorter GFF filer så start og stop codons ikke placeres i bunden
- [ ] Complement streng
- [x] Implementér bruger-option til at sætte threshold for minimum CDS længde de vil have rapport om (minimum: 30 - eller i hvert fald anbefaler vi ikke at gå længere ned!)
- [ ] Tillad input af gzipped input fasta
- [ ] Option for output fastafiler gzipped eller ej 
- [ ] Til aller sidst: opdater "Supplementary Note X" kommentarer i prediction script. 
- [ ] Optional og required argumenter i argparse! Se netstart 2!



### Scripts in development - A status
#### Data preprocessing
- [x] 1. /data_preprocessing/sort_taxonomy.py (Get taxonomic information for all organisms in dataset)
- [x] 2. /data_preprocessing/collect_genomic_information.py (merge taxonomic information for each organism with genomic statistical information and remove organisms with no family-level classification)
- [x] 3. /data_preprocessing/partition_genomes.py (partition genomes into test, val and train set based on pre-defined procedure)
- [x] 4. /data_preprocessing/extract_smaller_training_sets.py
- [x] 4. /data_preprocessing/simulate_reads.py (simulate reads of user-defined coverage and length on both template and complement strands)
- [x] 5. /data_preprocessing/process_reads_with_indels.py (processes datasets of reads with indel errors to extract necessary data)
- [x] 5. /data_preprocessing/process_reads_without_indels.py (processes datasets of reads without indel errors to extract necessary data)
- [x] 6. /data_preprocessing/postprocess_testset.py (Postprocess testset)
- [x] 6. /data_preprocessing/get_label_encodings.py (map class labels to 3d vectors; use for model that processes all 3 reading frames)
- [x] 7. /data_preprocessing/prepare_model_datasets.py (creates datasets specific for model input for each of the train and val splits)

#### Data analysis
- [ ] 1. /data_analysis/generate_taxonomical_trees.ipynb (Generate taxonomical trees in newick format along with partition annotations)
- [ ] 1. /data_analysis/plot_genome_statistics.ipynb (plot different genome statistics based on RefSeq annotations and genomes, for each data partition)
- [ ] 1. /data_analysis/get_testset_statistics.ipynb (get statistics for test set)
- [ ] 1. /data_analysis/check_testset_error_distributions.py (check error rates for each test set and check they are correct)

#### Modeling: Shared CRF models
- [ ] 1. /modeling/hyperparameter_tuning_shared_crf/hyperparameter_tuning_*.ipynb
- [ ] 2. /modeling/training_shared_crf/train_*.ipynb

#### Performance and benchmark
- [x] 1. /benchmark/predict/predict_with_fgs.ipynb (Predict with FGS)
- [x] 1. /benchmark/predict/predict_with_prodigal.ipynb (Predict with prodigal) 
- [x] 1. /benchmark/predict/predict_with_DeepCDS.py (Predict with DeepCDS)
- [x] 1. /benchmark/predict/predict_with_ESM2.py

- [ ] 2. /postprocess_preds/postprocess_model_predictions.ipynb (Postprocess testset)
- [ ] 2. /postprocess_preds/postprocess_fgs_predictions.ipynb (Postprocess testset)
- [ ] 2. /postprocess_preds/postprocess_prodigal_predictions.ipynb (Postprocess testset)


# Observationer træning: 
- [ ] 8M ESM-2 Frozen når næsten samem loss som ikke-frozen, men på væsentligt længere tid (115h/73h)
- [ ] Problemer med 150M ESM-2 S+I; validation loss non-coding spiker 
- [ ] 650M ESM-2 spiker ekstremt i coding sekvener for substitutionsmodel. Best val loss for 8M og 650M udgaverne er det samme, 35M udgaven har højere loss; Konklusion at 8M udgaven er "god nok", eller problem med hyperparametre for 35M og 650M udgaven?
- [ ] No errors 650M versionen har et lignende problem som substitutionsmodellen. 


### Project structure
Raw data transferred to ERDA

```
├── data/                           # Data directory
│   ├── processed_data/
│   |   ├── taxonomy                # Taxonomical distribution overview, processed
│   |   ├── dataset_information     # Taxonomical information & summary statistics for genomes
│   |   ├── genome_partitions       # Genome files distributed in train, val & test partitions
│   |   ├── simulated_reads         # Reads simulated with Mason
│   |   ├── processed_reads         # Processed reads with labelled positions and additional info
│   |   ├── model_data              # All data related to modeling
│   |   ├── predictions             # Predictions from DeepCDS, ablations and benchmark models
│   |   ├── testset_processed       # Testset processed to match format of prediction files #MOVE?
│   └── raw_data/
│   |   ├── genome_data             # Genome datasets (genome fasta files, gff3 annotation data)
│   |   ├── genome_data_info        # Genome datasets summary information
│   |   ├── taxonomy_data           # Taxonomical data from NCBI Taxonomy Database
├── models/                         # Trained models: NOT DEVELOPED: WANT THIS SEPERATELY PLACED HERE?
|
├── src_dev/                        # Source code
│   ├── data_preprocessing          #WRITE HERE!
│   ├── data_analysis
│   ├── modeling
│   ├── benchmark
|
├── .gitignore
├── pyproject.toml                  # Python project file
├── README.md                       # Project README
├── requirements.txt                # Project requirements: FILL OUT
├── requirements_dev.txt            # Development requirements: FILL OUT
```

### Info
````
>>> mapping_dict_to_class (subs & none)
{0: (0, 0, 0), 1: (0, 0, 1), 2: (0, 0, 2), 3: (0, 0, 3), 4: (0, 1, 0), 5: (0, 1, 1), 6: (0, 1, 2), 7: (0, 1, 3), 8: (0, 2, 0), 9: (0, 2, 1), 10: (0, 2, 3), 11: (0, 3, 0), 12: (0, 3, 1), 13: (1, 0, 0), 14: (1, 0, 1), 15: (1, 0, 2), 16: (1, 0, 3), 17: (1, 1, 0), 18: (1, 1, 2), 19: (1, 2, 0), 20: (1, 2, 3), 21: (1, 3, 0), 22: (2, 0, 0), 23: (2, 0, 1), 24: (2, 0, 3), 25: (2, 1, 0), 26: (2, 3, 0), 27: (3, 0, 0), 28: (3, 0, 1), 29: (3, 0, 2), 30: (3, 1, 0), 31: (3, 1, 1), 32: (3, 2, 0)}

>>> mapping_dict_to_class (indels + subs)
{0: (0, 0, 0), 1: (0, 0, 1), 2: (0, 0, 2), 3: (0, 0, 3), 4: (0, 0, 4), 5: (0, 0, 5), 6: (0, 1, 0), 7: (0, 1, 1), 8: (0, 1, 2), 9: (0, 1, 3), 10: (0, 1, 5), 11: (0, 2, 0), 12: (0, 2, 1), 13: (0, 2, 3), 14: (0, 2, 4), 15: (0, 3, 0), 16: (0, 3, 1), 17: (0, 3, 5), 18: (0, 4, 0), 19: (0, 4, 1), 20: (0, 4, 2), 21: (0, 4, 3), 22: (0, 4, 4), 23: (0, 5, 0), 24: (0, 5, 3), 25: (0, 5, 5), 26: (1, 0, 0), 27: (1, 0, 1), 28: (1, 0, 2), 29: (1, 0, 3), 30: (1, 0, 5), 31: (1, 1, 0), 32: (1, 1, 2), 33: (1, 2, 0), 34: (1, 2, 3), 35: (1, 3, 0), 36: (1, 5, 0), 37: (2, 0, 0), 38: (2, 0, 1), 39: (2, 0, 3), 40: (2, 0, 4), 41: (2, 0, 5), 42: (2, 1, 0), 43: (2, 3, 0), 44: (2, 3, 1), 45: (2, 4, 0), 46: (2, 5, 0), 47: (3, 0, 0), 48: (3, 0, 1), 49: (3, 0, 2), 50: (3, 0, 5), 51: (3, 1, 0), 52: (3, 1, 1), 53: (3, 2, 0), 54: (3, 5, 0), 55: (4, 0, 0), 56: (4, 0, 1), 57: (4, 0, 2), 58: (4, 0, 3), 59: (4, 0, 4), 60: (4, 1, 0), 61: (4, 2, 0), 62: (4, 3, 0), 63: (4, 4, 0), 64: (5, 0, 0), 65: (5, 0, 3), 66: (5, 0, 5), 67: (5, 3, 0), 68: (5, 5, 0)}
```

### Make public - notes
- [ ] Use ```pipreqs <dir>``` on both ```src_dev``` and ```src``` to get out required packages and versions (requirements and requirements_dev.txt). UPDATE THIS IN THE END!!! Does not take into account notebooks. 
- [ ] Write tool versions used for development + benchmark (Mason, FGS, MetaProdigal?)


Fjern nedenstående (få annoteringer):
 ['GCF_042926695.1', 'GCF_900635955.1', 'GCF_900636915.1', 'GCF_000026105.1']

find . -name "*GCF_042926695.1*" -exec rm -rf {} +
find . -name "*GCF_900635955.1*" -exec rm -rf {} +
find . -name "*GCF_900636915.1*" -exec rm -rf {} +
find . -name "*GCF_000026105.1*" -exec rm -rf {} +



Eksempler: 
Deletion: 
GCF_000007365.1_simulated_reads_template706|+|NC_004061.1|[[1,	DeepCDS	CDS	1	204	.	+	0	start=internal_region;end=indel_stop;group_id=group_1.0;indel_type=deletion
GCF_000007365.1_simulated_reads_template706|+|NC_004061.1|[[1,	DeepCDS	CDS	207	299	.	+	2	start=indel_start;end=internal_region;group_id=group_1.1;indel_type=deletion
GCF_000007365.1_simulated_reads_template706|+|NC_004061.1|[[1,	DeepCDS	uncertain_region	205	206	.	+	.	Note=Uncertain region: Frameshift gap between RF0 and RF2;overlapping_frames=0,2

Insertion:
GCF_000007365.1_simulated_reads_template528|+|NC_004061.1|[[2,	DeepCDS	CDS	2	220	.	+	1	start=internal_region;end=indel_stop;group_id=group_1.0;indel_type=insertion
GCF_000007365.1_simulated_reads_template528|+|NC_004061.1|[[2,	DeepCDS	CDS	222	299	.	+	2	start=indel_start;end=internal_region;group_id=group_1.1;indel_type=insertion
GCF_000007365.1_simulated_reads_template528|+|NC_004061.1|[[2,	DeepCDS	insertion	221	221	.	+	.	ID=insertion_GCF_000007365.1_simulated_reads_template528|+|NC_004061.1|[[2,_0

Start codon: 

Stop codon: 

CDS-eksempler uden interruptions:
GCF_000007365.1_simulated_reads_template498|+|NC_004061.1|[[1,	DeepCDS	CDS	1	300	.	+	0	start=internal_region;end=internal_region
GCF_000007365.1_simulated_reads_template500|+|NC_004061.1|[[3,	DeepCDS	CDS	3	98	.	+	2	start=internal_region;end=stop_codon
GCF_000007365.1_simulated_reads_template456|+|NC_004061.1|[[1,	DeepCDS	CDS	154	300	.	+	0	start=start_codon;end=internal_region



Deletion: 
- AAAGGNAAA (nukleotid; N deleted) og AXA (ukendt aminosyre i midten)

Insertion:
- Fjern inserted position, ret nemt


CCTCAATTCGAACTAGAGCAGCATATGGAACCAAAGATTAGAAGATCCTCAATAAGGAATTGCAAAGACAAAGAGATGGA...