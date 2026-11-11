# Real human gut metagenome: proof of concept

Do the CDS predicted on real gut metagenome reads look like known proteins? Every CDS predicted by DeepCDS (N, S,
S+I), MetaProdigal and FragGeneScanRs (Complete, Illumina 5, Illumina 10) on one real sample is searched against
UniRef90 with DIAMOND; the result is the benchmark table (`results/table_real_metagenome.tex`).
Nothing outside this directory (and its data directories) is modified.

## Sample
| | |
|---|---|
| Subject / ENA sample / run | MH0161 / ERS328874 / ERR321493 |
| Study | Le Chatelier et al. 2013 (Danish MetaHIT, PRJEB4336) |
| Metadata (curatedMetagenomicData `LeChatelierE_2013`) | Healthy, BMI 22.0, no current antibiotics, Denmark, feces |
| Sequencing | Illumina HiSeq 2000, paired-end 2x90 bp, 33,282,106 pairs |
| md5 (ENA) | `_1`: 0b678230496c602100c9ef4d4e6a3868, `_2`: 47d98501d7cde3d05f819fb96f077edd |

Selection: of the 39 Healthy samples with BMI 18.5-25, HiSeq 2000 only and all runs in ENA, the shallowest
(DeepCDS runtime). Age, sex and diabetes status not checked per sample (the cohort is non-diabetic adults).

Read counts: 33,282,106 pairs -> fastp 29,201,473 -> host filter 29,200,038 pairs -> R1 and R2 as separate reads,
**58,392,982 reads >= 61 bp** (the analysed population; 7,094 reads < 61 bp are not used).

## Method
1. **Prediction.** DeepCDS (N, S, S+I; `all_genomes` models, ESM-2 8M) on every read and its reverse complement;
   MetaProdigal (`-p meta`) and FragGeneScanRs (`-w 0`, `-t complete / illumina_5 / illumina_10`) with the benchmark
   commands. Only CDS >= 61 bp (>= 20 aa) are kept, for every tool, so that tools with different built-in minimum
   lengths are compared on the same CDS; a read is *detected* by a tool if it has >= 1 such CDS on either strand.
   All 362M CDS translate to >= 20 aa except 26 DeepCDS S+I CDS (18-19 aa after a predicted deletion).
2. **Translation.** Each CDS is translated from the read sequence with the GFF coordinates: table 11, except for
   MetaProdigal reads assigned a table-4 model. DeepCDS CDS split at a predicted indel are translated segment by
   segment and joined. FGS (Illumina 5 / 10) CDS are rebuilt with FGS's predicted indels (step 16, below).
   In-frame stop codons are kept as `X` and the CDS is flagged.
   **FGS Illumina indels (`16_fgs_illumina_indel_corrected.py`).** The FGS GFF coordinates leave out FGS's indel
   corrections, and FGS's own `.faa`/`.ffn` repeat the last codon of most CDS (98%), so neither is used as it is.
   FGS is rerun keeping its `.out` file (indel positions; the rerun GFF is identical to step 4's), and each CDS is
   rebuilt from the read: base removed at each predicted insertion, `N` added before each predicted deletion (also
   right after the reported end), reverse complement on the - strand. Every rebuilt CDS (all 128 chunks) equals
   FGS's `.ffn` sequence exactly or after removing its repeated final codon. The CDS, reads and read groups are
   unchanged; only the Illumina peptides are new (54.5M unique, searched as in step 12).
   **FGS last codon (`18_fgs_rf0_last_codon.py`).** FGS never lets a CDS reach the last base of a read: its end is
   1-3 bases short. With 3 bases left (the frame starting at the read's first base) it drops a complete codon that
   MetaProdigal and DeepCDS keep, so FGS had no 30 aa CDS. For all three FGS models such CDS are extended by those
   3 bases (after the indel correction for the Illumina modes), except when the + strand CDS already ends in a stop,
   the added codon is a stop, or FGS put a deletion right after the end. Extended: 32.7% (Complete), 32.7%
   (Illumina 5), 32.8% (Illumina 10) of FGS CDS. Every CDS not extended matches step 12 / 16 row by row (all 128
   chunks); only the 18.2M unique extended peptides were searched (same DIAMOND command). With this, FGS's length
   distribution matches the other tools (31-32% of CDS are 30 aa for every tool).
3. **DIAMOND.** DIAMOND 2.2.8 `blastp --sensitive --evalue 1e-3 --max-target-seqs 1` (BLOSUM62, gap 11/1) against
   UniRef90 release 2026_03 (121.5M clusters). A CDS *has a hit* if its peptide has any hit at E <= 1e-3.
   Identical peptides are searched once per chunk (83.7M unique peptides from 362M CDS).
4. **Control.** 157 random CDS per tool and chunk (seed 42 + chunk; 140,672 in total), residues shuffled, searched
   the same way: **0 hits**, so the hit rates are not inflated by spurious short-peptide hits. (For FGS Illumina
   the control was redrawn from the indel-corrected peptides in step 16, and for all FGS models from the step-18
   peptides: 60,288 sequences, also 0 hits; the total stays 140,672.)
5. **Read groups (table part B).** Each CDS is assigned the group of its read: which of DeepCDS S, MetaProdigal and
   FGS (Complete) detect the read. The hit rate of a tool in a group is over that tool's CDS on those reads.
   `14_agreement_all_models.py` does the same with the groups defined by all of DeepCDS N, S, S+I, MetaProdigal and
   FGS (Complete) (31 groups; `results/diamond_full_agreement_all_models.tsv`), and with all seven tools, i.e. also
   FGS (Illumina 5 / 10) with their indel-corrected CDS (120 non-empty groups; `..._all_tools.tsv`). FGS CDS are
   those of step 18. A DeepCDS model has the strictly highest hit rate in all 21 of the 31 groups where a DeepCDS
   model and MetaProdigal or FGS predict a CDS; over seven tools in 91 of 98 such groups (the exceptions have
   1-153 reads; six are ties, the other is 153 reads where FGS (Complete) has 46.4% vs 41.0% for S+I).

## Results
All tables as LaTeX: run `real_metagenome_tables.ipynb` (reads the result `.tsv` files, writes `results/tables/*.tex`:
per tool; read groups by DeepCDS S / MetaProdigal / FGS; by 5 tools; by 7 tools; CDS lengths).

**A** Per tool

| Tool | Reads with CDS (%) | Predicted CDS (M) | CDS with UniRef90 hit (%) |
|---|---|---|---|
| DeepCDS N | 80.1 | 47.5 | 83.7 |
| DeepCDS S | 81.1 | 48.1 | 84.0 |
| DeepCDS S+I | 85.4 | 51.6 | 80.5 |
| MetaProdigal | 88.5 | 51.7 | 75.1 |
| FGS (Complete) | 92.5 | 54.0 | 73.6 |
| FGS (Illumina 5) | 93.3 | 54.5 | 72.6 |
| FGS (Illumina 10) | 93.5 | 54.6 | 72.2 |

**B** Per read group (CDS with UniRef90 hit, %)

| CDS predicted by | Reads (%) | DeepCDS S | MetaProdigal | FGS (Complete) |
|---|---|---|---|---|
| All three | 78.0 | 84.7 | 80.8 | 81.6 |
| DeepCDS S + MetaProdigal | 0.8 | 66.0 | 58.0 | -- |
| DeepCDS S + FGS | 1.8 | 66.3 | -- | 54.3 |
| MetaProdigal + FGS | 8.5 | -- | 32.4 | 32.0 |
| DeepCDS S alone | 0.4 | 52.1 | -- | -- |
| MetaProdigal alone | 1.2 | -- | 15.6 | -- |
| FGS alone | 4.3 | -- | -- | 16.9 |

The remaining 5.0% of reads have no CDS from any of the three tools.
Median CDS length is 29 aa for every tool; 95.8-96.7% of CDS are >= 28 aa (`figures/cds_length_boxplots.*`).

Caveats (for footnotes):
- **FGS Illumina modes.** The hit rates use FGS's indel-corrected CDS (step 16). FGS predicts an indel in 2.8%
  (Illumina 5) and 4.3% (Illumina 10) of its CDS; on these the corrected protein has a hit less often than the
  uncorrected translation (17.7% vs 23.0% and 19.3% vs 24.1%, chunks 000-015), so most of FGS's predicted indels
  are not real on these reads. Overall (before step 18): 72.5% / 72.1% corrected vs 72.7% / 72.3% uncorrected
  (step-12 values in `results/diamond_full_cds_{per_tool,per_cell}_fgs_uncorrected.tsv`).
- **FGS end coordinates.** FGS leaves the last complete codon of a read out of its coordinates (on a 90 bp read
  in frame 1 it reports 1-87, MetaProdigal 1-90). Corrected in step 18 (as the benchmark corrects the end
  coordinate). Effect on hit rates: +0.1 points for each FGS model (73.5 -> 73.6, 72.5 -> 72.6, 72.1 -> 72.2);
  the step-16 values are in `results/diamond_full_cds_{per_tool,per_cell}_fgs_no_rf0_codon.tsv`.
- **DeepCDS in-frame stops.** 0.07% (N), 0.7% (S) and 1.1% (S+I) of DeepCDS CDS contain an in-frame stop.
  Excluding them, the hit rates change by at most 0.5 points.
- A hit is evidence of similarity to a known protein, not proof that the predicted frame and boundaries are
  correct, and a CDS without a hit may still be real (e.g. a gene without a homolog in UniRef90).

## Locations
- scripts: `src_dev/benchmark/real_metagenome/`
- results, figures, logs (ERDA): `FragmentPredictor/real_metagenome_poc/{results,figures,logs}/`; all `results/`,
  `figures/` and `logs/` paths in this README are relative to it
- working data (SCARB local disk): `/tmp/nrt204/real_metagenome_poc_work/` (`diamond_full/` holds `hits.tsv`,
  `hits_control.tsv` and the per-chunk CDS tables `chunks/cds_*.tsv.gz`; `diamond_fgs_indel/` the same for the
  indel-corrected FGS Illumina CDS, plus FGS's `.out` files; `diamond_fgs_rf0/` the same for the step-18 FGS CDS;
  UniRef90 DIAMOND database: was `/dev/shm/nrt204_uniref90/uniref90.dmnd`, removed by the server reboot of
  2026-10-30, rebuildable with `10a_download_uniref90.sh`)
- DeepCDS input / output (ERDA, read by HealthTech): `FragmentPredictor/real_metagenome_poc/{fasta,predictions}/`

## Pipeline (used for the table)
| Script | Step | Where |
|---|---|---|
| `01_qc_host_filter.sh` | fastp (defaults, paired) -> Bowtie2 vs GRCh38 no-alt; pairs kept only if both mates unmapped | SCARB |
| `02_make_fasta.py` | R1 and R2 as separate reads (`<spot>_1`, `<spot>_2`); >= 61 bp and < 61 bp files; DeepCDS input with each read + reverse complement | SCARB |
| `03_predict_with_DeepCDS_poc.py` + `03_predict_with_DeepCDS_healthtech.sh` | DeepCDS N, S, S+I on all reads | HealthTech |
| `04_predict_prodigal_fgs.sh` | MetaProdigal and FGS on the reads >= 61 bp (128 chunks in parallel) | SCARB |
| `07_detection_table.py` | per read and tool: >= 1 CDS >= 61 bp on either strand (`logs/07_detection_table_full.log`) | SCARB |
| `10a_download_uniref90.sh` | UniRef90 2026_03 download (md5 checked) -> DIAMOND database (database hash 9096220fb4180b4002b764a784f92690) | SCARB |
| `12_diamond_full_sample.py` | translation, DIAMOND vs UniRef90, control, per-tool and per-group hit rates (`logs/12_diamond_full_sample.log`; 5 h on 192 threads) | SCARB |
| `16_fgs_illumina_indel_corrected.py` | FGS Illumina CDS with FGS's indel corrections, DIAMOND, control; replaces the FGS Illumina rows of the step-12 results (`logs/16_fgs_illumina_indel_corrected.log`; 3.5 h) | SCARB |
| `18_fgs_rf0_last_codon.py` | FGS CDS (all three models) with the dropped last codon restored, DIAMOND on the extended peptides, control; replaces the FGS rows of the results (`logs/18_fgs_rf0_last_codon.log`; 1.5 h) | SCARB |
| `14_agreement_all_models.py` | hit rates per read group over five and over all seven tools (`logs/14_agreement_all_models.log`) | SCARB |
| `15_length_distributions.py` | read and predicted-CDS length distributions (`figures/length_distributions.{pdf,png}`, `results/length_distribution_{reads,cds}.tsv`) | SCARB |
| `17_cds_length_boxplots.py` | boxplots of the CDS lengths from the step-15 histogram (`figures/cds_length_boxplots.{pdf,png}`) | SCARB |
| `13_results_table.py` | benchmark table (`results/table_real_metagenome{.tex,_per_tool.tsv,_per_group.tsv}`) | SCARB |

`03_predict_with_DeepCDS_poc.py` is a copy of `src_dev/benchmark/predict_metagenome/predict_with_DeepCDS.py`;
the changes are listed in its docstring (input/output paths, `--sample`, per-run timing log, and removal of the
line `device = torch.device("cpu") #REMOVE ME AFTER TEST!`, which the original still contains).

## Superseded (not used for the table)
Earlier versions of the analysis, kept for reference:
- UHGG mapping categories: `00b_download_uhgg.sh`, `05_map_uhgg.sh`, `06_classify_mapping.py`, `08_analysis.py`,
  `09_spot_checks.py`; `results/upset_*.tsv`, `results/spot_check_*.tsv`, `figures/` (all but `length_distributions`). `13_results_table.py` only reads
  the total read count (`category == 'all'`) from `results/detection_rates.tsv`.
- DIAMOND on subsets: `10_diamond_check.py` (sampled CDS), `11_diamond_all_cds.py` (first 2M reads; its functions are
  imported by step 12); `results/diamond_hit_rates*.tsv`, `results/diamond_all_cds_*.tsv`.
