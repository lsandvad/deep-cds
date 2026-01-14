# DeepCDS
Project workspace for DeepCDS project

### TO DO
- [x] Make script to check number of reads in each length test set -> matches for all read lengths
- [ ] Define nt encoding full model (check d_model)
- [ ] Check up on substitution errors full model training convergence (running)
- [x] Simulate all test data
- [ ] Process all simulated reads (testset)
- [ ] Make plots nicer with scienceplots

### Data moved to ERDA:
- [x] Raw data ALL
- [ ] Processed data: Simulated reads
    - [x] Train/val
    - [ ] Test data: 30 bp reads
    - [ ] Test data: 60 bp reads
    - [ ] Test data: 75 bp reads
    - [ ] Test data: 100 bp reads
    - [ ] Test data: 150 bp reads
    - [x] Test data: 300 bp reads
    - [ ] Test data: 700 bp reads
    - [ ] Test data: 1000 bp reads
- [ ] Processed data: processed reads
    - [ ] Transfer train/val!
    - [ ] Test data: 30 bp reads (process on cluster)
    - [ ] Test data: 60 bp reads (process on cluster)
    - [ ] Test data: 75 bp reads (process on cluster)
    - [ ] Test data: 100 bp reads (process on cluster)
    - [ ] Test data: 150 bp reads (process on cluster)
    - [ ] Test data: 300 bp reads (process on cluster)
    - [ ] Test data: 700 bp reads (process on cluster)
    - [ ] Test data: 1000 bp reads (process on cluster)

### Scripts in development - A status
#### Data preprocessing
- [x] 1. /data_preprocessing/sort_taxonomy.py (Get taxonomic information for all organisms in dataset)
- [x] 2. /data_preprocessing/collect_genomic_information.py (merge taxonomic information for each organism with genomic statistical information and remove organisms with no family-level classification)
- [x] 3. /data_preprocessing/partition_genomes.py (partition genomes into test, val and train set based on pre-defined procedure)
- [x] 4. /data_preprocessing/extract_smaller_training_sets.py
- [x] 4. /data_preprocessing/simulate_reads.py (simulate reads of user-defined coverage and length on both template and complement strands)
- [ ] 5. /data_preprocessing/process_reads_with_indels.py (processes datasets of reads with indel errors to extract necessary data)
    - [x] Train and val data
    - [ ] Test data: 30 bp reads
    - [ ] Test data: 60 bp reads
    - [ ] Test data: 75 bp reads
    - [ ] Test data: 100 bp reads
    - [ ] Test data: 150 bp reads
    - [ ] Test data: 300 bp reads
    - [ ] Test data: 700 bp reads
    - [ ] Test data: 1000 bp reads
- [ ] 5. /data_preprocessing/process_reads_without_indels.py (processes datasets of reads without indel errors to extract necessary data)
    - [x] Train and val data
    - [ ] Test data: 30 bp reads
    - [ ] Test data: 60 bp reads
    - [ ] Test data: 75 bp reads
    - [ ] Test data: 100 bp reads
    - [ ] Test data: 150 bp reads
    - [ ] Test data: 300 bp reads
    - [ ] Test data: 700 bp reads
    - [ ] Test data: 1000 bp reads
- [ ] 6. /data_preprocessing/get_label_encodings.ipynb (map class labels to 3d vectors; use for model that processes all 3 reading frames)
- [ ] 7. /data_preprocessing/prepare_model_datasets.ipynb (creates datasets specific for model input for each of the train, val and test splits)

#### Data analysis
- [ ] 1. /data_analysis/generate_taxonomical_trees.ipynb (Generate taxonomical trees in newick format along with partition annotations)
- [ ] 2. /data_analysis/plot_genome_statistics.ipynb
- [ ] 2. /data_analysis/get_testset_statistics.ipynb
- [ ] 2. /data_analysis/plot_cds_lengths.ipynb

#### Modeling: Shared CRF models
- [ ] 1. /modeling/hyperparameter_tuning_shared_crf/hyperparameter_tuning_*.ipynb
- [ ] 2. /modeling/training_shared_crf/train_*.ipynb

#### Performance and benchmark
- [ ] 1. /benchmark/predict/model_predict_shared_crf.ipynb (Predict with models)
- [ ] 1. /benchmark/predict/predict_with_fgs.ipynb (Predict with FGS)
- [ ] 1. /benchmark/predict/predict_with_prodigal.ipynb (Predict with prodigal)
- [ ] 2. /postprocess_preds/postprocess_testset.ipynb (Postprocess testset)
- [ ] 2. /postprocess_preds/postprocess_model_predictions.ipynb (Postprocess testset)
- [ ] 2. /postprocess_preds/postprocess_fgs_predictions.ipynb (Postprocess testset)
- [ ] 2. /postprocess_preds/postprocess_prodigal_predictions.ipynb (Postprocess testset)



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


### Make public - notes
- [ ] Use ```pipreqs <dir>``` on both ```src_dev``` and ```src``` to get out required packages and versions (requirements and requirements_dev.txt). UPDATE THIS IN THE END!!! Does not take into account notebooks. 
- [ ] Write tool versions used for development + benchmark (Mason, FGS, MetaProdigal?)
