# DeepCDS: *FINAL TITLE*
DeepCDS is a deep learning-based model that predicts coding sequences (CDSs) in short prokaryotic DNA sequences. It also predicts start codon and stop codon positions. It can be used for prediction in both clean sequences, and sequences with sequencing errors (see `--error_model`). 

The model was developed based on 300bp long sequences, but tested on sequences in the length range from 60-1000bp. 

# Webserver 
*Link to and describe health tech server (if it will be hosted there)*

# Instructions for local use
## Setup
All DeepCDS source code required for inference has been written in Python. DeepCDS can be installed for local usage by first cloning this repository:
```
git clone https://github.com/lsandvad/deep-cds.git
cd deep-cds
```

The required packages can then be installed. Via pip:
```
pip install -e .
OR
pip install -r requirements.txt
```

If you want to setup a clean, isolated conda environment for DeepCDS, you can run:
```
conda env create -f environment.yml
conda activate deep-cds
```

## Usage
DeepCDS can be run via the command line by cloning this repository and installing the required packages as described in [Setup](#setup). 

To test the installation, you can run the following command from the project root:
```
python ./predict_with_deepcds.py -in ./data_example/test.fasta --error_model S
```

DeepCDS can be run to predict on your own data using the general command:
```
python ./predict_with_deepcds.py -in INPUT_FASTA -error_model ERROR_MODEL [optional arguments]
```

For a quick overview of all arguments, see [-Input](#input) below or run
```python ./predict_with_deepcds.py --help```

Please note that the DeepCDS prediction program uses the information stored in the /src, /models, and /configs directories. 

## Input
DeepCDS requires an input fasta file with the sequences to be predicted on, as well as which error model the user wants to use. Additionally, DeepCDS accepts a range of optional arguments:

| Input Argument                      | Description                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
|`-in`, `--input_fasta`       | Required: Input file in FASTA format. The allowed input alphabet is A, C, G, T, U and N (unknown). All the other letters will be treated as N. T and U are treated as equivalent. The input file can also be provided in gzipped format with a .gz extension.                                                                        |
|`--error_model` | Required: The type of sequence data DeepCDS was trained on based on presence of sequencing errors. Options are: `none` (DeepCDS (Full); trained on error-free data), `S` (DeepCDS S (Full); trained on sequences with substitution errors), `SI` (DeepCDS S+I (Full); trained on sequences with both substitution, insertion and deletion errors). Please note that this the choice of error model can notably influence your results. We recommend using `none` for complete genomic sequences without sequencing errors. |
|`--output` | Optional: The output file path and name witohut file format extension. Default: `<input_fasta_stem>_deepcds_predictions`. |
|`--compute_device` | Which hardware accelerator to use. Options are:  `cuda` (NVIDIA GPU), `mps` (Apple Silicon), or `cpu`. The program will automatically fall back to CPU if the requested device is unavailable. Default: `cuda`|
| `--batch_size`    | Optional: Specifies the number of samples to process together in a single pass during prediction. If you have limited memory, try a smaller batch size. Default value: `128`.                                     |
|`--min_cds_length` | Optional: Minimum length in base pairs for a predicted CDS sequences. We recommend not going below 30 base pairs as predictive performance below this threshold has not been evaluated. Default value: `60`|
|`--stride_aa` | Optional: The sliding window stride in codons for long sequences (how many codons the prediction window advances between each inference step). Smaller stride gives larger overlap between consecutive windows and may improve accuracy, but increases computation time. Default value: `50`.|
|`--gzip_output`| Optional: Specifies whether the output files should be gzipped (.gff.gz, .fna.gz, .faa.gz). Default value: `False`.|
|`--suppress_output_files`| Optional: Comma-separated list of output formats to suppress. Options: `gff`, `fna`, `faa`. For example, `--suppress_output_files fna,faa` will omit writing the CDS sequences to both nucleotide-level and amino acid-level fasta files and only write the annotations to a .gff file. Default: `None` (writes all output files).|

## Output formats
*skriv her; Skriv hvad output er! Forventer at vi outputter: GFF fil, 2x fasta filer med CDS på DNA-niveau og translaterede sekvenser*


# Noter og TODO til mig selv 
### TO DO opdateret 11. Maj
- [x] MOVE FGS AND METAPRODIGAL RAW PREDICTIONS!!! 
- [x] MOVE FGS AND METAPRODIGAL PROCESSED PREDICTIONS!!! 
- [x] Predict ART simulated reads with FGS og MetaProdigal
- [x] Postprocess ART simulated reads with FGS og MetaProdigal 
- [x] Push nye modeller til github før forsæt med implementering
- [x] Producer plot med fejlrater for art_modern simulerede reads
- [x] Skriv et "Supplementary Note" afsnit i overleaf om outputs (gff og fastaformater)
- [x] Gentræn alle 8M modeller "all_genomes"
- [ ] Få alle re-predictions ud "all_genomes"
- [ ] opdater alle resultater
    - [x] tests "error-free" opdateret alle steder
- [ ] Genlæs resultatsektion med nye predictions
- [ ] Tilføj ART simuleret read resultater
- [ ] Gentræn DeepCDS S+I (Full) på mindre datasæt 
- [ ] Få alle re-predictions ud "{100,200,400}_genomes"
- [ ] Implementering til inference (GitHub) (se to do features under)
    - [x] Implementering af script
    - [ ] Dokumentation i github; input og output beskrivelse 
    - [ ] Installation guideline og requirements
- [ ] Skriv diskussion
- [ ] Skriv abstract 

### Manuskript; mangler
- [ ] Abstract
- Resultater: 
    - [ ] "Findings from abalations..." ESM-2 ablations (træner stadig)
        - [ ] 100, 200, 400 genomer?
        - [ ] Træningsdata størrelse: Skriv at vi har undersøgt performance som funktion af træningsdata størrelsen (antal genomer) og det ikke giver så meget fra XX til XX… (enkelt sætning)
- [ ] Diskussion

### Implementering af script til prediction
- [x] Output fasta filer
- [x] Sorter GFF filer så start og stop codons ikke placeres i bunden
- [x] Complement streng
- [x] Implementér bruger-option til at sætte threshold for minimum CDS længde de vil have rapport om (minimum: 30 - eller i hvert fald anbefaler vi ikke at gå længere ned!)
- [x] Tillad input af gzipped input fasta
- [x] Option for output fastafiler gzipped eller ej 
- [x] Til aller sidst: opdater "Supplementary Note X" kommentarer i prediction script. 

### Scripts (clean-written: check boxes)
#### Data preprocessing
- [x] 1. /data_preprocessing/sort_taxonomy.py (Get taxonomic information for all organisms in dataset)
- [x] 2. /data_preprocessing/collect_genomic_information.py (merge taxonomic information for each organism with genomic statistical information and remove organisms with no family-level classification)
- [x] 3. /data_preprocessing/partition_genomes.py (partition genomes into test, val and train set based on pre-defined procedure)
- [x] 4. /data_preprocessing/extract_smaller_training_sets.py
- [x] 4. /data_preprocessing/simulate_reads.py (simulate reads of user-defined coverage and length on both template and complement strands)
- [x] 4. /data_preprocessing/simulate_reads_art_modern.py (simulate reads of user-defined coverage and length on both template and complement strands with art_modern for testing on another read simulator)
- [x] 5. /data_preprocessing/process_reads_with_indels.py (processes datasets of reads with indel errors to extract necessary data)
- [x] 5. /data_preprocessing/process_reads_without_indels.py (processes datasets of reads without indel errors to extract necessary data)
- [x] 5. /data_preprocessing/process_reads_from_art_modern.py (processes datasets of reads simulated with art_modern)
- [x] 6. /data_preprocessing/count_reads.py (Count reads per dataset for supplementary information, both train, val and test)
- [x] 6. /data_preprocessing/postprocess_testset.py (Postprocess testset)
- [x] 6. /data_preprocessing/get_label_encodings.py (map class labels to 3d vectors; use for model that processes all 3 reading frames)
- [x] 7. /data_preprocessing/prepare_model_datasets.py (creates datasets specific for model input for each of the train and val splits)

#### Data analysis
- [x] 1. /data_analysis/generate_taxonomical_trees.ipynb (Generate taxonomical trees in newick format along with partition annotations)
- [x] 1. /data_analysis/plot_genome_statistics.ipynb (plot different genome statistics based on RefSeq annotations and genomes, for each data partition)
- [x] 1. /data_analysis/get_testset_statistics.ipynb (get statistics for test set)
- [x] 1. /data_analysis/check_testset_error_distributions.py (check error rates for each test set and check they are correct)

#### Modeling scripts
- [ ] 1. /modeling/hyperparameter_tuning/hyperparameter_tuning_esm2.py
- [ ] 1. /modeling/hyperparameter_tuning/hyperparameter_tuning_full_model.py
- [ ] 2. /modeling/training/train_esm2.py
- [ ] 2. /modeling/training/train_full_model.py

#### Performance and benchmark UPDATE
- [x] 1. /benchmark/predict/predict_with_fgs.ipynb (Predict with FGS)
- [x] 1. /benchmark/predict/predict_with_prodigal.ipynb (Predict with prodigal) 
- [x] 1. /benchmark/predict/predict_with_DeepCDS.py (Predict with DeepCDS)
- [x] 1. /benchmark/predict/predict_with_ESM2.py

- [ ] 2. /postprocess_preds/postprocess_model_predictions.py (Postprocess testset)
- [ ] 2. /postprocess_preds/postprocess_fgs_predictions.ipynb (Postprocess testset)
- [ ] 2. /postprocess_preds/postprocess_prodigal_predictions.ipynb (Postprocess testset)


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