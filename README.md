# DeepCDS
Project workspace for DeepCDS project



### TO DO
- [x] Make script to check number of reads in each length test set -> matches for all read lengths
- [x] Adapt hyperparameter tuning scripts after fixed bug 
    - [ ] Full model
    - [ ] ESM 2
    - [x] Codon encoding
- [x] Simulate all test data
- [ ] Process all simulated reads (testset)
- [ ] Make plots nicer with scienceplots

### Scripts in development - A status
#### Data preprocessing
- [x] 1. /data_preprocessing/sort_taxonomy.py (Get taxonomic information for all organisms in dataset)
- [x] 2. /data_preprocessing/collect_genomic_information.py (merge taxonomic information for each organism with genomic statistical information and remove organisms with no family-level classification)
- [x] 3. /data_preprocessing/partition_genomes.py (partition genomes into test, val and train set based on pre-defined procedure)
- [x] 4. /data_preprocessing/extract_smaller_training_sets.py
- [x] 4. /data_preprocessing/simulate_reads.py (simulate reads of user-defined coverage and length on both template and complement strands)
- [ ] 5. /data_preprocessing/process_reads_with_indels.py (processes datasets of reads with indel errors to extract necessary data)
    - [ ] Train and val data
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
    - [x] Test data: 30 bp reads
    - [x] Test data: 60 bp reads
    - [x] Test data: 75 bp reads
    - [x] Test data: 100 bp reads
    - [x] Test data: 150 bp reads
    - [x] Test data: 300 bp reads
    - [x] Test data: 700 bp reads
    - [x] Test data: 1000 bp reads
- [x] 6. /data_preprocessing/get_label_encodings.py (map class labels to 3d vectors; use for model that processes all 3 reading frames)
- [x] 7. /data_preprocessing/prepare_model_datasets.py (creates datasets specific for model input for each of the train and val splits)
- [x] 8. /data_preprocessing/process_model_datasets_to_npy.py (process model datasets to be .npy files to be loaded more memory-efficiently for training)

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
    - [ ] Predict all on datasets with no sequencing errors 
    - [ ] Predict all on datasets with sequencing errors
- [x] 1. /benchmark/predict/predict_with_prodigal.ipynb (Predict with prodigal)
    - [ ] Predict all on datasets with no sequencing errors 
    - [ ] Predict all on datasets with sequencing errors
- [ ] 1. /benchmark/predict/model_predict_shared_crf.ipynb (Predict with models)
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
