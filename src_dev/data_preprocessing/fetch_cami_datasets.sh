# SCARB cluster mount point root - matches process_and_map_CAMI_testset.py's --data-root default
DATA_DIR="/tmp/nrt204/FragmentPredictor/data/raw_data/CAMI_datasets"
mkdir -p "$DATA_DIR" && cd "$DATA_DIR"

###LOAD ALL DATA INTO ABOVE DIR PATH
# Reference genomes used to build the community (ground-truth source)
wget https://s3.bi.denbi.de/swift/v1/cami/cami3_toydata/human-gut-toy/source_genomes.tar.gz

# Gold-standard read-to-genome mapping
wget https://s3.bi.denbi.de/swift/v1/cami/cami3_toydata/human-gut-toy/gsa_pooled_mapping.tsv.gz

# Sample 0 simulated short reads
wget https://s3.bi.denbi.de/swift/v1/cami/cami3_toydata/human-gut-toy/sample_0_reads.tar.gz

# Sample 0 BAM (read alignments to source genomes — replaces Mason's golden BAM role)
wget https://s3.bi.denbi.de/swift/v1/cami/cami3_toydata/human-gut-toy/sample_0_bam.tar.gz

tar -xzf source_genomes.tar.gz
tar -xzf sample_0_reads.tar.gz
tar -xzf sample_0_bam.tar.gz
gunzip gsa_pooled_mapping.tsv.gz