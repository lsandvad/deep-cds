# Homology-stratified evaluation

Does performance on the test set depend on sequence similarity to the training data? The error-free 300 bp test set
is stratified by sequence identity to the error-free training and validation data (the data DeepCDS N was trained
on), and CDS-level performance of DeepCDS N, MetaProdigal and FGS (Complete) is computed per identity bin
(manuscript Methods 2.5.2, Results 3.7, Figure 5, Supplementary Figures A9 and A21, Supplementary Table A37).

- **Protein level:** every labeled CDS fragment (> 60 bp) of the test set is translated and searched against all
  labeled train/val CDS fragments (MMseqs2 `-s 7.5 -e 10 --min-aln-len 20`); sensitivity per identity bin of the best hit.
- **Nucleotide level:** every test sequence, coding or not, is searched against all train/val sequences
  (`--search-type 3 --strand 1 -s 7.5 -e 10 --min-aln-len 60`); precision, sensitivity and F1 per identity bin, and
  the share of sequences overlapping annotated rRNA/tRNA genes.

## Run

```
setsid nohup bash run_audit.sh > run_audit.log 2>&1 &     # steps 0-10, several hours (mostly step 6)
```
then run `homology_audit_results.ipynb` (kernel `gene_prediction_env`) once with `EVALUE_CUTOFF = 10` (Figure 5) and
once with `EVALUE_CUTOFF = 1e-3` (Supplementary Figure A9). Figures and tables are written to `$RESULTS/figures` and `$RESULTS/tables`.

| Step | |
|---|---|
| `00_stage_data.sh` | copy inputs from the ERDA mount to local disk (`$WORK/staged`) |
| `01_extract_fragments.py` | translate test and train/val CDS fragments |
| `02_run_mmseqs.sh` | protein-level search |
| `03_assign_bins.py` | best hit and identity bin per test fragment |
| `04_evaluate_predictions.py` | TP per test fragment and tool; TP/FP/FN per genome |
| `05_extract_reads_nt.py` | test and train/val sequences as FASTA |
| `06_search_reads_nt.sh` | nucleotide-level search |
| `07_assign_reads_nt.py` | best hit per test sequence |
| `08_evaluate_reads_nt.py` | TP/FP/FN per test sequence and tool |
| `09_annotate_reads_rna.py` | test sequences overlapping rRNA/tRNA genes (> 60 bp, either strand) |
| `10_rna_gene_coverage.py` | share of each genome covered by rRNA/tRNA genes (all 1,125 genomes) |
| `benchmark_functions.py` | CDS matching and bootstrap functions, copied verbatim from `../../with_errors/cds_level_read_length.ipynb` |

## Paths and environments

- `$WORK` (default `/tmp/nrt204/homology_audit_work`, local disk, ~30 GB): staged inputs, FASTA files, MMseqs2 databases.
- `$RESULTS` (default `/tmp/nrt204/FragmentPredictor/data/processed_data/homology_audit`, ERDA): the result tables
  read by the notebook, and the figures and tables it makes.
- Python: `~/.conda/envs/gene_prediction_env`; MMseqs2 18.8cc5c: `~/.conda/envs/mmseqs_env`.
