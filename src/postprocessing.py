"""
DeepCDS postprocessing utilities.

Handles extraction of predicted CDS sequences from GFF output and translation
to amino acid sequences using the standard genetic code.

Note: The codon table here uses the standard bioinformatics conventions for
output files (*=stop, X=ambiguous). This is intentionally different from the
ESM-2 input encoding in deepcds_dataset.py (X=stop, <unk>=ambiguous).
"""

import contextlib
import gzip
from collections import defaultdict

from Bio import SeqIO


_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq):
    """Return the reverse complement of a nucleotide string."""
    return seq.translate(_RC_TABLE)[::-1]


# Standard genetic code for output files.
# Stop codons → '*', ambiguous/N-containing codons → 'X' (handled in translate_cds).
_CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


def translate_cds(nt_seq):
    """Translate a nucleotide sequence to amino acids using the standard genetic code.

    Codons containing N are translated as X. Stop codons are translated as *.
    Incomplete terminal codons (< 3 nt) are ignored.
    """
    aa = []
    nt_seq = str(nt_seq).upper()
    for i in range(0, len(nt_seq) - 2, 3):
        codon = nt_seq[i:i + 3]
        if 'N' in codon:
            aa.append('X')
        else:
            aa.append(_CODON_TABLE.get(codon, 'X'))
    return ''.join(aa)


def extract_cds_from_gff(fasta_path, gff_path, fna_path, faa_path):
    """Extract predicted CDS sequences from a GFF file and write nucleotide and
    protein FASTA output files.

    Each entry in the output files uses the GFF ID attribute as the sequence
    identifier (first word of the FASTA header), with source coordinates as a
    description. This allows direct cross-referencing between the three output
    files.

    For indel-interrupted CDS groups, fragments are merged into a single entry:
    insertions are removed, deletions are represented by an NNN gap (translated
    as X).

    Args:
        fasta_path: Path to the input nucleotide FASTA file.
        gff_path:   Path to the DeepCDS GFF output file.
        fna_path:   Output path for the nucleotide CDS FASTA (.fna).
        faa_path:   Output path for the protein CDS FASTA (.faa).
    """
    import sys

    _open = gzip.open if str(gff_path).endswith(".gz") else open
    sequences = SeqIO.index(fasta_path, "fasta")

    ungrouped = []
    groups = defaultdict(list)
    entry_order = []   # ('ungrouped', index) or ('grouped', key), in GFF order
    seen_groups = set()

    try:
        with _open(gff_path, "rt") as gff_f:
            for line in gff_f:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 9 or fields[2] != "CDS":
                    continue

                attrs = dict(item.split("=", 1) for item in fields[8].split(";"))

                # CDS fragments interrupted by indels have group_id in format "group_X.Y"
                if "group_id" in attrs:
                    group_base = attrs["group_id"].rsplit(".", 1)[0]
                    key = (fields[0], group_base)
                    groups[key].append((fields, attrs))
                    if key not in seen_groups:
                        seen_groups.add(key)
                        entry_order.append(("grouped", key))
                else:
                    entry_order.append(("ungrouped", len(ungrouped)))
                    ungrouped.append((fields, attrs))

        with contextlib.ExitStack() as stack:
            fna_f = stack.enter_context(_open(fna_path, "wt")) if fna_path else None
            faa_f = stack.enter_context(_open(faa_path, "wt")) if faa_path else None

            for entry_type, key in entry_order:

                if entry_type == "ungrouped":
                    fields, attrs = ungrouped[key]
                    seq_name = fields[0]
                    start, end, strand = int(fields[3]), int(fields[4]), fields[6]
                    fwd_seq = sequences[seq_name].seq[start - 1 : end]
                    cds_seq = fwd_seq.reverse_complement() if strand == "-" else fwd_seq
                    cds_id = attrs.get("ID", f"{seq_name}_{start}_{end}_{strand}")
                    header = f">{cds_id}"
                    if fna_f is not None:
                        fna_f.write(f"{header}\n{cds_seq}\n")
                    if faa_f is not None:
                        faa_f.write(f"{header}\n{translate_cds(cds_seq)}\n")

                else:  # grouped
                    seq_name, group_id = key
                    strand = groups[key][0][0][6]
                    # Sort by fragment number (.0, .1, …) — biological 5'→3' order for
                    # both strands (assigned in write_gff in RC/forward position order).
                    members = sorted(groups[key],
                                     key=lambda x: int(x[1]["group_id"].rsplit(".", 1)[1]))
                    indel_type = members[0][1]["indel_type"]

                    fragments = []
                    for f, _ in members:
                        fwd_seq = sequences[seq_name].seq[int(f[3]) - 1 : int(f[4])]
                        fragments.append(
                            str(fwd_seq.reverse_complement()) if strand == "-"
                            else str(fwd_seq)
                        )

                    # Insertion is removed
                    if indel_type == "insertion":
                        merged_seq = "".join(fragments)
                    # NNN gap represents the deleted region (translates as X)
                    elif indel_type == "deletion":
                        merged_seq = "NNN".join(fragments)

                    cds_id = members[0][1].get("ID", f"{seq_name}_{group_id}")
                    header = f">{cds_id}"
                    if fna_f is not None:
                        fna_f.write(f"{header}\n{merged_seq}\n")
                    if faa_f is not None:
                        faa_f.write(f"{header}\n{translate_cds(merged_seq)}\n")

    except ValueError as e:
        print(f"Error processing GFF file: {e}")
        sys.exit(1)
