# DeepCDS: *FINAL TITLE*
DeepCDS is a deep learning-based model that predicts coding sequences (CDSs) in short prokaryotic DNA sequences. It also predicts start codon and stop codon positions. It can be used for prediction in both clean sequences, and sequences with sequencing errors. 

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

The required packages can then be installed. 

#### Via pip
```
pip install -r requirements.txt
```

#### Via conda environment
If you want to set up a clean and isolated conda environment for DeepCDS, you can run:
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

For a quick overview of all arguments, see [Input Arguments](#input-arguments) below or run
```python ./predict_with_deepcds.py --help```.

Please note that the DeepCDS prediction program uses the information stored in the /src, /models, and /configs directories. 

## Input Arguments
DeepCDS requires an input fasta file with the sequences to be predicted on, as well as which error model the user wants to use. Additionally, DeepCDS accepts a range of optional arguments:

| Input Argument                      | Description                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
|`-in`, `--input_fasta`       | Required: Input file in FASTA format. The allowed input alphabet is A, C, G, T, U and N (unknown). All the other letters will be treated as N. T and U are treated as equivalent. The input file can also be provided in gzipped format with a .gz extension.                                                                        |
|`--error_model` | Required: The type of sequence data DeepCDS was trained on based on presence of sequencing errors. Options are: `none` (DeepCDS (Full); trained on error-free data), `S` (DeepCDS S (Full); trained on sequences with substitution errors), `SI` (DeepCDS S+I (Full); trained on sequences with both substitution, insertion and deletion errors). Please note that the choice of error model can notably influence your results. We recommend using `none` for complete genomic sequences without sequencing errors. |
|`--output` | Optional: The output file path and name without file format extension. Default: `<input_fasta_stem>_deepcds_predictions` (written to the current working directory). |
|`--compute_device` | Which hardware accelerator to use. Options are: `cuda` (NVIDIA GPU), `mps` (Apple Silicon), `cpu`, and `auto` (selects the best available device in order cuda &rarr; mps &rarr; cpu). The program will automatically fall back to CPU if the requested device is unavailable. Default: `auto`|
| `--batch_size`    | Optional: Specifies the number of samples to process together in a single pass during prediction. If you have limited memory, try a smaller batch size. Default value: `128`.                                     |
|`--min_cds_length` | Optional: Minimum length in base pairs for a predicted CDS sequences. We recommend not going below 30 base pairs as predictive performance below this threshold has not been evaluated. Default value: `60`|
|`--stride_aa` | Optional: The sliding window stride in codons for long sequences (how many codons the prediction window advances between each inference step). Smaller stride gives larger overlap between consecutive windows and may improve accuracy, but increases computation time. Default value: `50`.|
|`--gzip_output`| Optional: Specifies whether the output files should be gzipped (.gff.gz, .fna.gz, .faa.gz). Default value: `False`.|
|`--suppress_output_files`| Optional: Comma-separated list of output formats to suppress. Options: `gff`, `fna`, `faa`. For example, `--suppress_output_files fna,faa` will omit writing the CDS sequences to both nucleotide-level and amino acid-level fasta files and only write the annotations to a .gff file. See [Output formats](#output-formats) for a description of the output files. Default: `None` (writes all output files).|

## Output formats
The output is provided as three files: a .gff file with the CDS annotations (including start codon and stop codon positions), a .fna file with the predicted CDS sequences, and a .faa file with the predicted CDS sequences translated into the corresponding amino acid sequence. 

### .gff notes
#### Feature types
- `CDS`: A coding sequence region annotation. 
- `start_codon`: Start codon annotation. Please note that the beginning of a CDS annotation in short sequence fragments does not necessarily equal a start codon position, as DeepCDS can predict CDS regions that are only internal regions of a protein, only the start of a protein, or the end of a protein. 
- `stop_codon`: Stop codon annotation. Please note that the end of a CDS annotation in short sequence fragments does not necessarily a stop codon position, as DeepCDS can predict CDS regions that are only internal regions of a protein, only the start of a protein, or the end of a protein. 
- `insertion`: A specific inserted nucleotide position that has been directly identified. This is a special case where the exact insertion site is known, in contrast to the `uncertain_region` feature which marks ambiguous positions when the exact site cannot be determined. CDS fragments flanking an `insertion` share a `group_id` attribute. This feature type is only predicted with `--error_model SI`.
- `uncertain_region`: Marks ambiguous nucleotide positions between CDS fragments interrupted by a predicted insertion or deletion error, where the exact indel position cannot be directly determined. CDS fragments flanking an `uncertain_region` share a `group_id` attribute. This feature type is only predicted with `--error_model SI`.

#### Attribute information
Attributes are provided as a list of tag-value pairs. Each pair is separated by a semicolon. 
- `ID`: Unique ID for annotation. 
- `start`: the state that the given feature started in, for example `start_codon` or `internal_region`. 
- `end`: the state that the given feature ended in, for example `stop_codon` or `internal_region`. 
- `group_id`: CDS regions interrupted by an insertion or deletion error are split into two or more CDS feature annotations, and share a common `group_id` attribute in order to connect CDS fragments that belong to the same coding sequence. This attribute is only provided with `--error_model SI`.
- `indel_type`: Provided together with `group_id`. Marks the kind of sequencing error predicted (either `insertion` or `deletion`). This attribute is only provided with `--error_model SI`.
- `overlapping_frames`: Marks which reading frames the two CDS fragments flanking a `type=uncertain_region` are placed in. This attribute is only provided with `--error_model SI`.
- `Note`: Any additional notes related to the given annotation. 

### .fna notes
Fasta file containing the predicted CDS sequences. In cases where a deletion error has been predicted, the missing region in the merged CDS sequence is represented as an "NNN" codon.  

### .faa notes
Fasta file containing the translated CDS sequences (using the standard prokaryotic translation table; NCBI genetic code 11). In cases where a deletion error has been predicted, the missing region in the merged CDS sequence is represented as an "NNN" codon that is translated as "X". Furthermore, all codons with one or more unknown nucleotide positions are translated as "X", and stop codons are denoted as "*".


# Noter og TODO til mig selv 
### TO DO opdateret 22. Maj
- [ ] Træn codon encoding-only modeller
    - [ ] None
    - [ ] S -> KØRER
    - [ ] SI -> KØRER
- [ ] Predict på codon encoding only modeller (typical error datasæt + art_modern?)
- [ ] Resultatplot for codon encoding
- [x] Implementering til inference (GitHub)
- [ ] HealthTech server implementering?
- [ ] Skriv cover letter
- [ ] Opdater kommentarer mm. i scripts
- [ ] Supplementary Tables i Excel: saml, giv et navn etc.

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

#### Performance and benchmark
- [x] 1. /benchmark/predict/predict_with_fgs.ipynb (Predict with FGS)
- [x] 1. /benchmark/predict/predict_with_prodigal.ipynb (Predict with prodigal) 
- [x] 1. /benchmark/predict/predict_with_DeepCDS.py (Predict with DeepCDS)
- [x] 1. /benchmark/predict/predict_with_ESM2.py

- [ ] 2. /postprocess_preds/postprocess_model_predictions.py (Postprocess testset)
- [ ] 2. /postprocess_preds/postprocess_fgs_predictions.ipynb (Postprocess testset)
- [ ] 2. /postprocess_preds/postprocess_prodigal_predictions.ipynb (Postprocess testset)

- [x] 3. /eval/without_errors/start_stop_coodn_evaluation.ipynb (start and stop cdon identification performance; test sets without erros)
- [x] 3. /eval/without_errors/codon_level_read_length.ipynb (codon-level performance; measured as MCC; test sets without errors)
- [ ] 3. /eval/without_errors/organisms_families_gc_content.ipynb (different analyses measured on different phylogenetic groups and across GC content intervals; test sets without errors)
- [ ] 3. /eval/without_errors/cds_level_read_length.ipynb (CDS-level performance across all test set read lengths; test sets without errors)

- [x] 3. /eval/with_errors/start_stop_codon_evaluation.ipynb (start and stop cdon identification performance; test sets with errors)
- [x] 3. /eval/with_errors/codon_level_read_length.ipynb (codon-level performance; measured as MCC; test sets with errors)
- [ ] 3. /eval/with_errors/organisms_families_gc_content.ipynb (different analyses measured on different phylogenetic groups and across GC content intervals; test sets with errors)
- [ ] 3. /eval/with_errors/cds_level_read_length.ipynb (CDS-level performance across all test set read lengths; test sets with errors)
- [ ] 3. INDEL SCRIPT
- [ ] 3. PLOT_METRICS_VS_ERROR_RATE SCRIPT

