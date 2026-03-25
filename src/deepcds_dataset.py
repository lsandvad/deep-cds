"""
DeepCDS Dataset Classes and Data Processing Functions

This module contains the dataset classes and data processing utilities for the DeepCDS system:
- SeqDataset: PyTorch Dataset for sequence data
- Data encoding and preprocessing functions
- Nucleotide to amino acid translation
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# Standard genetic code for translation
GENETIC_CODE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': 'X', 'TAG': 'X',
    'TGT': 'C', 'TGC': 'C', 'TGA': 'X', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}


def translate_nucleotide_to_amino_acid(sequence):
    """
    Translate a nucleotide sequence into the corresponding amino acid sequence.

    Args:
        sequence (str): A nucleotide sequence.

    Returns:
        str: The amino acid sequence translated from the nucleotide input.
             Each codon is mapped to its corresponding amino acid,
             with stop codons represented by "X".
    """
    sequence = str(sequence)

    # Adjust length to be divisible by 3
    remainder = len(sequence) % 3
    if remainder:
        sequence = sequence[:-remainder]

    # Direct list comprehension for efficiency
    return ''.join(GENETIC_CODE.get(sequence[i:i+3], "<unk>")
                   for i in range(0, len(sequence), 3))


def one_hot_encode(sequence):
    """
    One-hot encode nucleotide sequences in a matrix format of 4 rows (A, C, G, T)
    and len(sequence) columns. Optimized for memory efficiency.

    Args:
        sequence (str): A nucleotide sequence.

    Returns:
        torch.Tensor: One-hot encoded sequence with shape (4, len(sequence)).
    """
    seq_len = len(sequence)
    encoding = torch.zeros(4, seq_len, dtype=torch.float32)

    # Vectorized approach using byte conversion
    seq_bytes = torch.frombuffer(bytearray(sequence.encode('ascii')), dtype=torch.uint8)

    # Map nucleotides to indices: A(65)->0, C(67)->1, G(71)->2, T(84)->3
    for nuc, idx in ((65, 0), (67, 1), (71, 2), (84, 3)):
        mask = seq_bytes == nuc
        encoding[idx, mask] = 1.0

    # N and other characters remain as zeros (no explicit handling needed)
    return encoding


def process_nt_sequences_to_codons(nt_sequences, max_aa_len):
    """
    Convert nucleotide sequences from dimension (4, nucleotide_seq_len) to dimension (max_aa_len, 12) format
    by grouping every 3 nucleotides (1 codon) together. Vectorized for efficiency.

    Args:
        nt_sequences: List of tensors with shape (4, nucleotide_seq_len)
        max_aa_len: Maximum amino acid sequence length

    Returns:
        List of tensors with shape (max_aa_len, 12)
    """
    # Stack all sequences for batch processing
    stacked = torch.stack(nt_sequences)  # (batch, 4, nt_len)

    # Reshape to group codons: (batch, 4, max_aa_len, 3)
    batch_size = stacked.shape[0]
    reshaped = stacked.view(batch_size, 4, max_aa_len, 3)

    # Permute to (batch, max_aa_len, 4, 3) then flatten last two dims
    transposed = reshaped.permute(0, 2, 1, 3)  # (batch, max_aa_len, 4, 3)
    formatted = transposed.reshape(batch_size, max_aa_len, 12)  # (batch, max_aa_len, 12)

    # Return as list of tensors (unbind is efficient)
    return list(formatted.unbind(0))


class SeqDataset(Dataset):
    """
    PyTorch Dataset for sequence data with nucleotide and amino acid encodings.

    Args:
        nt_encodings_rf0 (list): List of nucleotide codon encodings (max_aa_len, 12) for reading frame 0
        aa_encodings_rf0 (BatchEncoding): Tokenized amino acid encodings (dict) for reading frame 0
        nt_encodings_rf1 (list): List of nucleotide codon encodings (max_aa_len, 12) for reading frame 1
        aa_encodings_rf1 (BatchEncoding): Tokenized amino acid encodings (dict) for reading frame 1
        nt_encodings_rf2 (list): List of nucleotide codon encodings (max_aa_len, 12) for reading frame 2
        aa_encodings_rf2 (BatchEncoding): Tokenized amino acid encodings (dict) for reading frame 2
        read_name: Unique read identifier for genome accession
    """

    def __init__(self, nt_encodings_rf0, aa_encodings_rf0,
                 nt_encodings_rf1, aa_encodings_rf1,
                 nt_encodings_rf2, aa_encodings_rf2,
                 read_name):

        self.nt_encodings_rf0 = nt_encodings_rf0
        self.aa_encodings_rf0 = aa_encodings_rf0

        self.nt_encodings_rf1 = nt_encodings_rf1
        self.aa_encodings_rf1 = aa_encodings_rf1

        self.nt_encodings_rf2 = nt_encodings_rf2
        self.aa_encodings_rf2 = aa_encodings_rf2

        self.read_name = read_name

    def __getitem__(self, idx):
        item = {
            'nt_encodings_rf0': torch.as_tensor(self.nt_encodings_rf0[idx], dtype=torch.float32),
            'aa_encodings_rf0': {key: val[idx] for key, val in self.aa_encodings_rf0.items()},

            'nt_encodings_rf1': torch.as_tensor(self.nt_encodings_rf1[idx], dtype=torch.float32),
            'aa_encodings_rf1': {key: val[idx] for key, val in self.aa_encodings_rf1.items()},

            'nt_encodings_rf2': torch.as_tensor(self.nt_encodings_rf2[idx], dtype=torch.float32),
            'aa_encodings_rf2': {key: val[idx] for key, val in self.aa_encodings_rf2.items()},

            'read_name': self.read_name[idx]
        }
        return item

    def __len__(self):
        return len(self.read_name)


def encode_data(processed_samples_df, max_aa_len, tokenizer=None, esm2_model_name="facebook/esm2_t6_8M_UR50D"):
    """
    Encode data samples to fit model input format.

    Args:
        processed_samples_df (DataFrame): Dataframe with input dataset.
        max_aa_len (int): Maximum amino acid input length; max_len without special tokens (CLS and EOS)
        tokenizer (AutoTokenizer, optional): Specific ESM tokenizer. If None, will be created.
        esm2_model_name (str): Name of the ESM-2 model for tokenizer initialization.

    Returns:
        SeqDataset: Dataset with data formatted to fit model input.
    """

    max_len = max_aa_len + 2  # Add CLS and EOS tokens

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            esm2_model_name,
            do_lower_case=False)

    # Initialize dictionaries to hold encodings for each reading frame
    encodings_nt = {}
    encodings_aa = {}
    max_nt_len = max_aa_len * 3


    # Process data from each RF separately
    for rf in ["rf0", "rf1", "rf2"]:
        # ==== Nucleotide sequence processing ====
        # Trim sequence starts according to reading frame (RF1 offsets by 1, RF2 offsets by 2 relative to start of read)
        if rf == "rf0":
            processed_samples_df[f"{rf}_seq_nt"] = processed_samples_df["read"].apply(lambda seq: seq) 
        elif rf == "rf1":
            processed_samples_df[f"{rf}_seq_nt"] = processed_samples_df["read"].apply(lambda seq: seq[1:])
        elif rf == "rf2":
            processed_samples_df[f"{rf}_seq_nt"] = processed_samples_df["read"].apply(lambda seq: seq[2:])

        # ==== Amino acid sequence processing ====
        processed_samples_df[f"{rf}_seq_aa"] = processed_samples_df[f"{rf}_seq_nt"].apply(
            lambda seq: translate_nucleotide_to_amino_acid(seq)
        )
        aa_sequences = processed_samples_df[f"{rf}_seq_aa"].tolist()

        # Tokenize with strict length control
        aa_encodings_rf = tokenizer(
            aa_sequences,
            padding="max_length",      # Pad all to max_length
            max_length=max_len,        # CLS + max_aa_len AA + EOS
            truncation=True,           # Cut longer sequences
            return_tensors="pt",       # Return PyTorch tensors
        )

        encodings_aa[rf] = aa_encodings_rf

        # ==== Nucleotide sequence processing continued ====
        # Pad or truncate nucleotide sequences to max_nt_len (3 * max_aa_len), which ensures full codon coverage in all three reading frames. 
        # See "Supplementary Note X. Inference on sequence ends" 
        processed_samples_df[f"{rf}_seq_nt"] = processed_samples_df[f"{rf}_seq_nt"].apply(
            lambda seq: seq + 'N' * (max_nt_len - len(seq)) if len(seq) < max_nt_len else seq[:max_nt_len])

        nt_sequences = [one_hot_encode(seq) for seq in processed_samples_df[f"{rf}_seq_nt"]]

        # Process nt_sequences to codon-based format (3*4=12, max_nt_len/3)
        nt_encodings_rf = process_nt_sequences_to_codons(nt_sequences, max_aa_len)
        encodings_nt[rf] = nt_encodings_rf

        read_name = processed_samples_df["read_name"]

    dataset = SeqDataset(
        encodings_nt["rf0"], encodings_aa["rf0"],
        encodings_nt["rf1"], encodings_aa["rf1"],
        encodings_nt["rf2"], encodings_aa["rf2"],
        read_name)

    return dataset
