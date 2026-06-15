# DeepCDS: *Ab initio* coding sequence prediction in prokaryotic short reads
DeepCDS is a deep learning-based model that predicts coding sequences (CDSs) in short prokaryotic DNA sequences, including start codon and stop codon positions. It can be used for prediction in both clean sequences, and sequences with sequencing errors. 

The model was developed based on 300bp long sequences, but tested on sequences in the sequence length range from 60-1000bp. 

# Webserver 
For smaller datasets, the DeepCDS 1.0 prediction server is available for use [here](https://services.healthtech.dtu.dk/services/DeepCDS-1.0/).

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
|`--error_model` | Required: The type of sequence data DeepCDS was trained on based on presence of sequencing errors. Options are: `none` (DeepCDS N; trained on error-free data), `S` (DeepCDS S; trained on sequences with substitution errors), `SI` (DeepCDS S+I; trained on sequences with both substitution, insertion and deletion errors). Please note that the choice of error model can notably influence your results. We recommend using `none` for complete genomic sequences without sequencing errors. |
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