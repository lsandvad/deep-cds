# deep-cds
Project workspace for DeepCDS project


### Scripts in development - A status
#Data preprocessing
DONE 1. /data_preprocessing/sort_taxonomy.py (Get taxonomic information for all organisms in dataset)
DONE 2. /data_preprocessing/collect_genomic_information.py (merge taxnomic information for each organism with genomic statistical information and remove organisms with no family-level classification)
DONE 3. /data_preprocessing/partition_genomes.py (partition genomes into test, val and train set based on pre-defined procedure)
4. /data_preprocessing/extract_smaller_training_sets.py
DONE 4. scripts/data_preprocessing/simulate_reads.ipynb (simulate reads of user-defined coverage and length on both template and complement strands)
DONE 5. scripts/data_preprocessing/process_reads_with_indels.py (processes datasets of reads with indel errors to extract necessary data)
DONE 5. scripts/data_preprocessing/process_reads_without_indels.py (processes datasets of reads without indel errors to extract necessary data)
6. scripts/data_preprocessing/get_label_encodings.ipynb (map class labels to 3d vectors; use for model that processes all 3 reading frames)
7. scripts/data_preprocessing/prepare_model_datasets.ipynb (creates datasets specific for model input for each of the train, val and test splits) 
7. scripts/data_preprocessing/prepare_model_datasets_shared.ipynb (creates datasets specific for model input for each of the train, val and test splits) 

#Data analysis
1. /data_analysis/generate_taxonomical_trees.ipynb (Generate taxonomical trees in newick format along with partition annotations)
2. /data_analysis/plot_genome_statistics.ipynb
2. /data_analysis/get_testset_statistics.ipynb
2. /data_analysis/plot_cds_lengths.ipynb

#Modeling: Shared CRF models
1. /modeling/hyperparameter_tuning_shared_crf/hyperparameter_tuning_*.ipynb
2. /modeling/training_shared_crf/train_*.ipynb

#Performance and benchmark
1. /benchmark/predict/model_predict_shared_crf.ipynb                #Predict with models
1. /benchmark/predict/predict_with_fgs.ipynb                        #Predict with FGS
1. /benchmark/predict/predict_with_prodigal.ipynb                   #Predict with prodigal
2. /postprocess_preds/postprocess_testset.ipynb                     #Postprocess testset 
2. /postprocess_preds/postprocess_model_predictions.ipynb           #Postprocess testset 
2. /postprocess_preds/postprocess_fgs_predictions.ipynb             #Postprocess testset 
2. /postprocess_preds/postprocess_prodigal_predictions.ipynb        #Postprocess testset 

