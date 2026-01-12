#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import optuna
from tqdm import tqdm
import gc
import psutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef
from transformers import AutoTokenizer, AutoModel
from functools import lru_cache
from torch.amp import GradScaler, autocast

#Clear the GPU memory cache
torch.cuda.empty_cache()
pd.options.mode.chained_assignment = None  # Suppress the warning globally

import wandb
import os
import json

from torchcrf import CRF # pip install pytorch-crf
import ast
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


#Clear the GPU memory cache
torch.cuda.empty_cache()

#Configure CUDA memory allocations (helps manage fragmentation in the GPU memory)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


# In[2]:


#Change input directory path for ERDA
input_data_dir_path = "/tmp/nrt204/FragmentPredictor2/data/processed_data/model_data/single_rf_crf/model_with_errors/datasets_model" #TEST ON SCARB CLUSTER
#input_data_dir_path = None #Local tests

if input_data_dir_path == None: 
    input_data_dir_path = "../../../data/processed_data/model_data/single_rf_crf/model_with_errors/datasets_model"


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device_type = device.type  # "cuda", "mps", or "cpu"

if device_type == "mps":
    num_workers_cpu = 0 #Adjust if not using mps
    pin_memory = False #Turn true if using CUDA
elif device_type == "cuda":
    num_workers_cpu = 4
    pin_memory = True

print("Device type:", device_type)

#Model choice
esm2_model = "facebook/esm2_t6_8M_UR50D"
#esm2_model = "facebook/esm2_t33_650M_UR50D"
esm2_model_abbr = esm2_model.split("/")[-1].split("_UR")[0]
project_name = esm2_model_abbr

#dir in wandb to place run
wandb_project_name = "optuna_single_rf_crf_errors" #MODIFY!

#Make sure dir to store model exists
os.makedirs(f"../../../data/processed_data/model_data/single_rf_crf/model_with_errors/models/", exist_ok=True)

max_aa_len = 100
max_len = max_aa_len + 2 #Add CLS and EOS tokens


# In[4]:


def clear_memory():
    #Memory clean up function
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# In[5]:


genetic_code = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '<unk>', 'TAG': '<unk>',
    'TGT': 'C', 'TGC': 'C', 'TGA': '<unk>', 'TGG': 'W',
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
    The function takes a nucleotide sequence and translates it into the corresponding amino acid sequence. 

    Args:
        sequence (str): A nucleotide sequence.

    Returns:
        (str): A string representing the amino acid sequence translated from the 
               nucleotide input. Each codon is mapped to its corresponding amino acid,
               with stop codons represented by "<unk>".
    """

    sequence = str(sequence)

    # More efficient length adjustment
    remainder = len(sequence) % 3
    if remainder:
        sequence = sequence[:-remainder]

    # Direct list comprehension is fastest
    return ''.join(genetic_code.get(sequence[i:i+3], "<mask>") 
                   for i in range(0, len(sequence), 3))


# In[6]:


def one_hot_encode(sequence):
    """
    One-hot encode nucleotide sequences in a matrix format of 4 rows (A, C, G, T)
    and len(sequence) columns.

    Args:
        sequence (str): a nucleotide sequence. 

    Returns: 
        encoding (tensor): one-hot encoded sequence. 
    """

    #Define the mapping of nucleotides to indices
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    #Create an empty one-hot encoding tensor
    encoding = torch.zeros(4, len(sequence))

    #Convert the sequence to a tensor of indices for efficient indexing
    indices = torch.tensor(
        [mapping[char] for char in sequence if char in mapping], dtype=torch.long)

    #Use advanced indexing to set the appropriate positions to 1
    positions = torch.arange(len(sequence))[[char in mapping for char in sequence]]
    encoding[indices, positions] = 1
    #For 'N', we do nothing, so the corresponding column remains all zeros

    return torch.tensor(encoding)


# In[7]:


def process_nt_sequences_to_codons(nt_sequences):
    """
    Convert nucleotide sequences from (4, 300) to (12, 100) format
    by grouping every 3 nucleotides (1 codon) together.

    Args:
        nt_sequences: List of tensors with shape (4, 300)

    Returns:
        List of tensors with shape (12, 100)
    """
    processed_sequences = []

    for seq in nt_sequences:
        # seq has shape (4, 300)
        # Reshape to (4, 100, 3) to group every 3 nucleotides
        seq_reshaped = seq.view(4, 100, 3)

        # Transpose to (100, 4, 3) then reshape to (100, 12)
        seq_transposed = seq_reshaped.transpose(0, 1)  # (100, 4, 3)
        seq_formatted = seq_transposed.reshape(100, 12)  # (100, 12)

        processed_sequences.append(seq_formatted)

    return processed_sequences


# In[8]:


class SeqDataset(torch.utils.data.Dataset):
    """
    Optimized dataset class with reduced memory overhead.
    """
    def __init__(self, nt_encodings, aa_encodings, labels, seq_desc):
        self.nt_encodings = nt_encodings
        self.aa_encodings = aa_encodings
        self.labels = labels  
        self.seq_desc = seq_desc

    def __getitem__(self, idx):
        item = {
            'aa_encodings': {
                key: val[idx] for key, val in self.aa_encodings.items()
            },
            'nt_encodings': torch.as_tensor(self.nt_encodings[idx], dtype=torch.float32),
            'labels': torch.from_numpy(self.labels[idx]) if isinstance(self.labels[idx], np.ndarray) else torch.tensor(self.labels[idx], dtype=torch.float32),
            'seq_desc': self.seq_desc[idx]
        }
        return item

    def __len__(self):
        return len(self.labels)


# In[9]:


def encode_data(processed_samples_df, max_len, tokenizer=None):
    """ 
    Encode data samples to fit model input format. 

    Args:
        processed_samples_df (dataframe): Dataframe with input dataset.

    Returns:
        dataset (dict): nested dictionary with data formatted to fit model input.
    """

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/esm2_t6_8M_UR50D",
            do_lower_case=False,
        )

    #Label processing
    if isinstance(processed_samples_df["aa_labels"].iloc[0], str):
        processed_samples_df["aa_labels"] = processed_samples_df["aa_labels"].apply(eval)

    #Convert to numpy arrays more efficiently
    label_arrays = [np.array(x, dtype=np.int8) for x in processed_samples_df["aa_labels"]]

    #Nucleotide sequence processing
    nt_sequences = [one_hot_encode(seq) for seq in processed_samples_df["nt_seq"]] #[300, 4, num_seqs]

    # Process nt_sequences to codon-based format (3*4=12, 100)
    nt_encodings = process_nt_sequences_to_codons(nt_sequences)

    #Amino acid sequence processing
    aa_sequences = processed_samples_df["aa_seq"].tolist()

    #Tokenize with strict length control
    aa_encodings = tokenizer(
        aa_sequences,
        padding="max_length",      #Pad all to max_length
        max_length=max_len,        #CLS + 100 AA + EOS = 102
        truncation=True,           #Cut longer sequences
        return_tensors="pt",       #Return PyTorch tensors
    )

    pad_positions = max_len - 2
    padded_labels = np.full((len(label_arrays), pad_positions), 9, dtype=np.int8)

    for i, arr in enumerate(label_arrays):
        length = min(len(arr), pad_positions)
        padded_labels[i, :length] = arr[:length]

    seq_descriptions = processed_samples_df["aa_desc"].tolist()

    dataset = SeqDataset(nt_encodings, aa_encodings, padded_labels, seq_descriptions)

    return dataset, list(set(seq_descriptions))


# In[10]:


def balance_training_samples(dataset_df):
    """ 
    Down-sample sequences so that there is equally many sequences that are: 
        1. Fully non-coding
        2. Fully coding
        3. Contains a transition (due to start/end of CDS or an indel error causing a frameshift)

    Args:
        dataset_df (df): Dataframe with input dataset.

    Returns: 
        balanced_df (df): Dataframe with balanced representation of sequence types.
    """

    def classify_sequence(label_seq):
        """ 
        Classify a sequence based on its label composition.

        Args:
            label_seq (array-like): Sequence of labels (e.g., [0, 0, 0], [1, 1, 1], [0, 1, 0], etc.)

        Returns:
            str: 
                - "flat_0" if all labels are 0 (fully non-coding)
                - "flat_1" if all labels are 1 (fully coding)
                - "transition" if the sequence contains a mix of labels (i.e., contains a transition)
        """
        unique_labels = set(label_seq)  #Get unique label values in the sequence
        unique_labels = {int(x) for x in unique_labels if str(x).isdigit()}  #Ensure labels are integers
        if unique_labels == {0}:
            return "flat_0"  #Fully non-coding
        elif unique_labels == {1}:
            return "flat_1"  #Fully coding
        else:
            return "signal_change"  #Contains a signal change (transition caused by indels or start/stop codons, substitution errors) (mixed labels)

    #Apply classification
    dataset_df["seq_type"] = dataset_df["aa_labels"].apply(classify_sequence)

    #Count how many transition sequences there are
    n_transition = len(dataset_df[dataset_df["seq_type"] == "signal_change"])

    #Sample that many from each of the flat types (or fewer if you want more imbalance)
    flat_0_sampled = dataset_df[dataset_df["seq_type"] == "flat_0"].sample(n=min(n_transition*4, len(dataset_df[dataset_df["seq_type"] == "flat_0"])), random_state=42)
    flat_1_sampled = dataset_df[dataset_df["seq_type"] == "flat_1"].sample(n=min(n_transition*4, len(dataset_df[dataset_df["seq_type"] == "flat_1"])), random_state=42)

    #Keep all transition sequences
    transition_set = dataset_df[dataset_df["seq_type"] == "signal_change"]

    #Combine them
    balanced_df = pd.concat([transition_set, flat_0_sampled, flat_1_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    print("Data samples, balanced: ", balanced_df.shape[0])

    return balanced_df


# In[11]:


def load_and_process_data(max_len):
    """
    Main function that loads and processes all data efficiently.
    """
    # Load data
    train_set = pd.read_csv(
        f"{input_data_dir_path}/train_small.csv.gz", #MODIFY
        index_col=None, 
        compression="gzip"
    )
    val_set = pd.read_csv(
        f"{input_data_dir_path}/val.csv.gz", #MODIFY
        index_col=None, 
        compression="gzip"
    )

    val_set = val_set.sample(frac=0.02, random_state=42)

    seq_type_desc_fracs = (val_set['aa_desc'].value_counts(normalize=True)).to_dict()

    print("Training data samples: ", train_set.shape[0])
    print("Validation data samples: ", val_set.shape[0])
    print("Distribution of sequence types in validation set: ", seq_type_desc_fracs)

    # Create tokenizer once and reuse
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esm2_t6_8M_UR50D",
        do_lower_case=False,
    )

    # Process training data
    balanced_train = balance_training_samples(train_set)
    train_data, sequence_types = encode_data(balanced_train, max_len, tokenizer)

    # Clear intermediate data to save memory
    del balanced_train

    # Process validation data
    balanced_val = balance_training_samples(val_set)
    val_data, _ = encode_data(balanced_val, max_len, tokenizer)

    # Clear intermediate data
    del balanced_val

    return train_data, val_data, sequence_types, seq_type_desc_fracs


# # Model framework with CRF

# In[12]:


class SequenceEncoder(nn.Module):
    """
    Sequence encoder using a pretrained ESM-2 model.
    """
    def __init__(self,
                 esm2_model,
                 dropout_rate_1): 
        super(SequenceEncoder, self).__init__()

        # Load pretrained ESM-2 model for amino acid sequences
        self.pretrained_model_aa = AutoModel.from_pretrained(esm2_model)

        # Freeze early layers of ESM to save memory and compute
        for i, layer in enumerate(self.pretrained_model_aa.encoder.layer):
            if i < len(self.pretrained_model_aa.encoder.layer) // 2:
                for param in layer.parameters():
                    param.requires_grad = False

        # Dropout layer for regularization before transformer head
        self.dropout_1 = nn.Dropout(dropout_rate_1)

    def forward(self, x_aa, attention_mask_aa):
        """
        """
        # Extract features from pretrained ESM-2 model
        features_aa = self.pretrained_model_aa(x_aa, attention_mask=attention_mask_aa)

        # Get last hidden state: [batch_size, tokens, hidden_size]
        sequence_output_aa = features_aa['last_hidden_state']

        # Remove CLS and EOS token embeddings: [batch_size, aa_seq_len, hidden_size]
        sequence_output_aa = sequence_output_aa[:, 1:-1, :]

        # Apply dropout before transformer head
        embeddings_aa = self.dropout_1(sequence_output_aa)

        # Remove CLS/EOS from attention mask for transformer head
        attention_mask_trimmed = attention_mask_aa[:, 1:-1]

        return embeddings_aa, attention_mask_trimmed


# In[13]:


class TransformerEncoderBlock(nn.Module):
    """
    Neural network module that applies a Transformer encoder block to encoded sequences and adds a linear layer to fit CRF input.

    Args:
        hidden_size (int): The dimensionality of the input and output features for the Transformer encoder.
        num_layers (int): Number of Transformer encoder layers to stack.
        n_attention_heads (int): Number of attention heads in each Transformer encoder layer.
        dropout_rate (float): Dropout rate applied after normalization and within the encoder layers.
        act_function (str or Callable): Activation function to use in the feedforward network of the encoder layers.
        num_labels (int): Number of output classes 

    Attributes:
        encoder (nn.TransformerEncoder): Stacked Transformer encoder layers.
        classifier (nn.Linear): Linear layer mapping the encoder output to class logits.
        norm (nn.LayerNorm): Layer normalization applied after the encoder.
        dropout (nn.Dropout): Dropout layer applied after normalization.
        layers (int): Number of encoder layers.
    """
    def __init__(self,
                 hidden_size,
                 num_layers,
                 n_attention_heads,
                 dropout_rate_encoder,
                 dropout_rate_final,
                 act_function,
                 num_labels):
        super().__init__()

        hidden_size_merged = hidden_size + 12 #12 for codon one-hot encoded; hidden_size for amino acid representation from ESM2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size_merged,
            nhead=n_attention_heads,
            dim_feedforward=4*hidden_size_merged,
            dropout=dropout_rate_encoder,
            activation=act_function
        )

        self.layers = num_layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size_merged, num_labels)
        self.norm = nn.LayerNorm(hidden_size_merged)
        self.dropout_final = nn.Dropout(dropout_rate_final)


    def forward(self, encoded_seqs_nt, encoded_embeddings_aa, trimmed_attention_mask):
        """
        Forward pass through the Transformer encoder block and linear classifier.

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, hidden_size]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]

        Returns:
            torch.Tensor of logits [batch_size, seq_len, num_labels]
        """

        # Original transformer processing
        if self.layers > 0:

            combined_codon_and_aa_embeddings = torch.cat([encoded_embeddings_aa, encoded_seqs_nt], dim=-1)

            combined_codon_and_aa_embeddings = combined_codon_and_aa_embeddings.permute(1, 0, 2)  # [seq_len, batch, hidden + 3*4]
            attention_mask_transformer = ~trimmed_attention_mask.bool()

            combined_codon_and_aa_embeddings = self.encoder(combined_codon_and_aa_embeddings, src_key_padding_mask=attention_mask_transformer)

            combined_codon_and_aa_embeddings = combined_codon_and_aa_embeddings.permute(1, 0, 2)  # [batch, seq_len, hidden]

        combined_codon_and_aa_embeddings = self.norm(combined_codon_and_aa_embeddings)
        combined_codon_and_aa_embeddings = self.dropout_final(combined_codon_and_aa_embeddings)

        logits = self.linear(combined_codon_and_aa_embeddings)  # [batch, seq_len, num_labels]

        return logits


# In[14]:


class LinearChainCRF(nn.Module):
    """
    Neural network module that adds a CRF layer for structured prediction.
    """
    def __init__(self, num_labels, crf_init_weight):
        super().__init__()
        self.crf = CRF(num_tags=num_labels, 
                       batch_first=True)

        # Define legal transitions as adjacency list
        self.legal_transitions = {
            0: {0, 2, 4},  
            1: {1, 3, 5},  
            2: {1, 5},     
            3: {0, 2, 4},  
            4: {1, 3, 5},  
            5: {0, 2, 4}   
        }

        self.constraint_mask = torch.ones_like(self.crf.transitions, dtype=torch.bool)
        self._create_constraint_mask()

        with torch.no_grad():
            self.crf.transitions[~self.constraint_mask] = -crf_init_weight  # forbidden
            self.crf.transitions[self.constraint_mask] = crf_init_weight   # allowed

    def _create_constraint_mask(self):
        """Create mask from legal_transitions dictionary."""
        # Start with all transitions forbidden
        self.constraint_mask.fill_(False)

        # Mark legal transitions as True
        for from_label, to_labels in self.legal_transitions.items():
            for to_label in to_labels:
                self.constraint_mask[from_label, to_label] = True

    def forward(self, logits, attention_mask, labels=None):
        """
        Forward pass with CRF layer.
        """
        # Apply constraints before forward pass (this maintains gradients for learnable params)

        if labels is not None:
            # Training mode: compute CRF loss
            crf_mask = attention_mask.bool()
            loss = -self.crf(logits, labels, mask=crf_mask, reduction='mean')

            return {'loss': loss, 'logits': logits}
        else:
            # Inference mode: decode best sequence
            crf_mask = attention_mask.bool()
            predictions = self.crf.decode(logits, mask=crf_mask)
            return {'predictions': predictions, 'logits': logits}



# In[15]:


class CDSPredictor(nn.Module):
    """
    Model architecture for CDS prediction using a pretrained ESM-2 model and a Transformer head with CRF.
    """
    def __init__(self,
                 esm2_model,
                 num_layers,
                 n_attention_heads,
                 dropout_rate_1,
                 dropout_rate_2,
                 dropout_rate_3,
                 act_function,
                 crf_init_weight): 
        super(CDSPredictor, self).__init__()

        self.sequence_encoder = SequenceEncoder(
            esm2_model,
            dropout_rate_1)

        self.TransformerEncoderBlock = TransformerEncoderBlock(
            hidden_size=self.sequence_encoder.pretrained_model_aa.config.hidden_size,
            num_layers=num_layers,
            n_attention_heads=n_attention_heads,
            dropout_rate_encoder=dropout_rate_2,
            dropout_rate_final=dropout_rate_3,
            act_function=act_function,
            num_labels=6)

        self.CRF = LinearChainCRF(num_labels=6,
                                  crf_init_weight=crf_init_weight)

        self.relu = nn.ReLU()


    def forward(self, encoded_seqs_nt, x_aa, attention_mask_aa, labels=None):
        """
        Forward pass through the model with CRF support.

        Args:
            x_aa (tensor): Amino acid token encodings [batch_size, seq_len].
            attention_mask_aa (tensor): Attention mask [batch_size, seq_len].
            labels (tensor, optional): True labels [batch_size, aa_seq_len] for CRF training.

        """

        encoded_embeddings_aa, trimmed_attention_mask = self.sequence_encoder(x_aa, attention_mask_aa)

        logits = self.TransformerEncoderBlock(
            encoded_seqs_nt=encoded_seqs_nt,
            encoded_embeddings_aa=encoded_embeddings_aa,
            trimmed_attention_mask=trimmed_attention_mask)

        output = self.CRF(
            logits=logits,
            attention_mask=trimmed_attention_mask,
            labels=labels)

        return output


# In[16]:


def initialize_model(device, num_layers, n_attention_heads, dropout_rate_1, dropout_rate_2, dropout_rate_3, act_function, crf_init_weight):
    """
    Initialize the model and move it to the specified device.
    Args:
        device_type (str): The device to use for computation ("cuda", "mps", or "cpu").
    Returns:
        device (torch.device): The device being used.
        model (nn.Module): The initialized model.
    """

    #if device_type == "mps":
    #    device = torch.device(device_type if torch.mps.is_available() else "cpu")
    #elif device_type == "cuda":
    #    device = torch.device(device_type if torch.cuda.is_available() else "cpu")   

    print("Running on: ", device, flush = True)

    model = CDSPredictor(esm2_model=esm2_model,
                         num_layers = num_layers,
                         n_attention_heads = n_attention_heads,
                         dropout_rate_1 = dropout_rate_1, 
                         dropout_rate_2 = dropout_rate_2,
                         dropout_rate_3 = dropout_rate_3,
                         act_function = act_function,
                         crf_init_weight = crf_init_weight)
    model.to(device)

    if device == "cuda":
        print(f"Memory Allocated after loading model: {torch.cuda.memory_allocated(device) / 1024**3} GB")

    return model


def print_model_dimensions(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[17]:


def calculate_sequence_accuracy_metrics(true_labels_list, predictions_list, sequence_types_list=None):
    """
    Calculate sequence-level accuracy metrics including sensitivity, specificity, and MCC for overall and for each label specifically.
    Also calculates sequence type-specific metrics if sequence_types_list is provided.

    Args:
        true_labels_list: List of numpy arrays, each containing true labels for one sequence
        predictions_list: List of numpy arrays, each containing predictions for one sequence  
        sequence_types_list: Optional list of sequence types/descriptions corresponding to each sequence

    Returns:
        results (dict): Dictionary containing accuracy metrics including overall MCC and type-specific metrics
    """
    if len(true_labels_list) != len(predictions_list):
        raise ValueError("Mismatch in number of sequences")

    if sequence_types_list is not None and len(sequence_types_list) != len(true_labels_list):
        raise ValueError("Mismatch in number of sequence types")

    classes = [0, 1, 2, 3, 4, 5]
    total_sequences = len(true_labels_list)

    # Use more memory-efficient approach
    perfect_sequences = 0
    high_accuracy_sequences = 0
    sequence_accuracies = []

    # For sequence type-specific metrics
    type_specific_data = {}
    if sequence_types_list is not None:
        unique_types = list(set(sequence_types_list))
        for seq_type in unique_types:
            type_specific_data[seq_type] = {
                'true_labels': [],
                'predictions': [],
                'sequence_accuracies': [],
                'perfect_sequences': 0,
                'high_accuracy_sequences': 0,
                'total_sequences': 0
            }

    # Process sequences in batches to save memory
    batch_size = 1000  # Process 1000 sequences at a time
    all_true_labels = []
    all_predictions = []

    for i in range(0, total_sequences, batch_size):
        batch_true = true_labels_list[i:i+batch_size]
        batch_pred = predictions_list[i:i+batch_size]
        batch_types = sequence_types_list[i:i+batch_size] if sequence_types_list else None

        batch_accuracies = []
        batch_all_true = []
        batch_all_pred = []

        for idx, (true_labels, predictions) in enumerate(zip(batch_true, batch_pred)):
            true_labels = np.asarray(true_labels)
            predictions = np.asarray(predictions)

            # Add to overall arrays
            batch_all_true.extend(true_labels.tolist())
            batch_all_pred.extend(predictions.tolist())

            # Calculate accuracy
            accuracy = (true_labels == predictions).mean()
            batch_accuracies.append(accuracy)

            if accuracy == 1.0:
                perfect_sequences += 1
            if accuracy > 0.9:
                high_accuracy_sequences += 1

            # Handle type-specific data
            if batch_types is not None:
                seq_type = batch_types[idx]
                type_data = type_specific_data[seq_type]

                # Add to type-specific arrays
                type_data['true_labels'].extend(true_labels.tolist())
                type_data['predictions'].extend(predictions.tolist())
                type_data['sequence_accuracies'].append(accuracy)
                type_data['total_sequences'] += 1

                if accuracy == 1.0:
                    type_data['perfect_sequences'] += 1
                if accuracy > 0.9:
                    type_data['high_accuracy_sequences'] += 1

        sequence_accuracies.extend(batch_accuracies)
        all_true_labels.extend(batch_all_true)
        all_predictions.extend(batch_all_pred)

        # Clear batch data
        del batch_true, batch_pred, batch_accuracies, batch_all_true, batch_all_pred
        if batch_types is not None:
            del batch_types

    # Convert to numpy arrays
    all_true_labels = np.array(all_true_labels)
    all_predictions = np.array(all_predictions)

    # Calculate overall MCC efficiently
    try:
        overall_mcc = matthews_corrcoef(all_true_labels, all_predictions)
        if np.isnan(overall_mcc):
            overall_mcc = 0.0
    except:
        overall_mcc = 0.0

    # Calculate per-class accuracies more efficiently
    class_accuracies = {}
    for cls in classes:
        mask = all_true_labels == cls
        if mask.sum() > 0:
            class_accuracies[f'acc_class_{cls}'] = (all_predictions[mask] == all_true_labels[mask]).mean()
        else:
            class_accuracies[f'acc_class_{cls}'] = 0.0

    # Build results
    results = {
        'fraction_perfect_sequences': perfect_sequences / total_sequences,
        'fraction_high_accuracy_sequences': high_accuracy_sequences / total_sequences,
        'overall_mcc': overall_mcc,
        'accuracy': np.mean(sequence_accuracies),
        **class_accuracies
    }

    # Calculate type-specific metrics
    if sequence_types_list is not None:
        type_metrics = {}
        for seq_type, type_data in type_specific_data.items():
            if type_data['total_sequences'] > 0:
                # Convert type-specific data to numpy arrays
                type_true = np.array(type_data['true_labels'])
                type_pred = np.array(type_data['predictions'])

                # Calculate type-specific MCC
                try:
                    type_mcc = matthews_corrcoef(type_true, type_pred)
                    if np.isnan(type_mcc):
                        type_mcc = 0.0
                except:
                    type_mcc = 0.0

                # Calculate type-specific metrics
                type_metrics[f'{seq_type}_mcc'] = type_mcc
                type_metrics[f'{seq_type}_accuracy'] = np.mean(type_data['sequence_accuracies'])
                type_metrics[f'{seq_type}_fraction_perfect'] = type_data['perfect_sequences'] / type_data['total_sequences']
                type_metrics[f'{seq_type}_fraction_high_accuracy'] = type_data['high_accuracy_sequences'] / type_data['total_sequences']

                # Calculate per-class accuracies for this type
                for cls in classes:
                    mask = type_true == cls
                    if mask.sum() > 0:
                        type_metrics[f'{seq_type}_acc_class_{cls}'] = (type_pred[mask] == type_true[mask]).mean()
                    else:
                        type_metrics[f'{seq_type}_acc_class_{cls}'] = 0.0

        # Add type-specific metrics to results
        results.update(type_metrics)

    return results


# ## Track loss of different sequence types, adjust distribution of training samples accordingly

# In[18]:


class CategoricalLossTracker:
    """ 
    A class to track losses for different categories during training.
    """
    def __init__(self, categories):
        self.categories = categories
        self.losses = {cat: [] for cat in categories} # Create empty list for each category

    def update(self, category, loss):
        #Extract average loss for the category in a batch
        self.losses[category].append(loss.item())

    def get_metrics(self):
        # Calculate the mean loss for each category
        return {cat: np.mean(losses) for cat, losses in self.losses.items()}


def create_weighted_sampler(dataset, type_weights):
    """
    Create a weighted sampler based on sequence types and their weights.
    """
    # Get sequence types for all samples in dataset
    seq_types = [dataset[i]['seq_desc'] for i in range(len(dataset))]

    # Create sample weights based on type_weights
    sample_weights = [type_weights.get(seq_type, 1.0) for seq_type in seq_types]

    # Create WeightedRandomSampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )

    return sampler


# In[19]:


def log_evaluation_metrics(epoch, train_avg_loss, val_avg_loss, best_val_loss, tracker, sequence_metrics, val_times_counter, sequence_types, sequence_type_weights):
    # Build the print statement with type-specific metrics
    print_parts = [
        f"---Evaluation {val_times_counter}---\n",
        f"Train Loss: {train_avg_loss:.4f}\t\t",
        f"Val Loss: {val_avg_loss:.4f}\t\t"
    ]

    # Add overall metrics
    print_parts.extend([
        f"Overall MCC: {sequence_metrics.get('overall_mcc', 0):.4f}\t\t",
        f"Overall Accuracy: {sequence_metrics.get('accuracy', 0):.4f}\n"
    ])

    # Add loss metrics for each sequence type
    for seq_type in sequence_types:
        loss_val = tracker.get_metrics().get(seq_type, 0)
        print_parts.append(f"Val Loss {seq_type}: {loss_val:.4f}\t\t")

    print_parts.append("\n")

    # Add type-specific MCC and accuracy metrics
    for seq_type in sequence_types:
        type_mcc = sequence_metrics.get(f'{seq_type}_mcc', 0)
        type_acc = sequence_metrics.get(f'{seq_type}_accuracy', 0)
        if type_mcc != 0 or type_acc != 0:  # Only show if we have data for this type
            print_parts.extend([
                f"MCC {seq_type}: {type_mcc:.4f}\t\t",
                f"Acc {seq_type}: {type_acc:.4f}\t\t"
            ])

    # Print all metrics
    print("".join(print_parts), flush=True)

    # Build wandb logging dictionary
    wandb_log = {
        "epoch": epoch + 1,
        "train_loss": train_avg_loss,
        "val_loss": val_avg_loss,

        # Overall sequence metrics
        "val_fraction_perfect_sequences": sequence_metrics.get('fraction_perfect_sequences', 0),
        "val_fraction_high_accuracy_sequences": sequence_metrics.get('fraction_high_accuracy_sequences', 0),
        "val_overall_mcc": sequence_metrics.get('overall_mcc', 0),
        "val_accuracy": sequence_metrics.get('accuracy', 0)}

    val_loss_weighted_average = 0

    # Add loss metrics for each sequence type
    for seq_type in sequence_types:
        wandb_log[f"val_loss_{seq_type}"] = tracker.get_metrics().get(seq_type, 0)
        val_loss_weighted_average += tracker.get_metrics().get(seq_type, 0) * sequence_type_weights[seq_type]

    # Add type-specific MCC and accuracy metrics
    for seq_type in sequence_types:
        # MCC metrics
        type_mcc = sequence_metrics.get(f'{seq_type}_mcc', 0)
        if type_mcc != 0:  # Only log if we have data
            wandb_log[f"val_mcc_{seq_type}"] = type_mcc

        # Accuracy metrics
        type_acc = sequence_metrics.get(f'{seq_type}_accuracy', 0)
        if type_acc != 0:  # Only log if we have data
            wandb_log[f"val_accuracy_{seq_type}"] = type_acc

        # Perfect and high accuracy sequence fractions
        type_perfect = sequence_metrics.get(f'{seq_type}_fraction_perfect', 0)
        if type_perfect != 0:
            wandb_log[f"val_fraction_perfect_{seq_type}"] = type_perfect

        type_high_acc = sequence_metrics.get(f'{seq_type}_fraction_high_accuracy', 0)
        if type_high_acc != 0:
            wandb_log[f"val_fraction_high_accuracy_{seq_type}"] = type_high_acc

    wandb_log[f"val_loss_weighted_average"] = val_loss_weighted_average
    wandb_log[f"best_val_loss"] = best_val_loss

    # Log to wandb
    wandb.log(wandb_log)

    return val_times_counter + 1


# In[20]:


def adjust_train_sample_distribution(train_data, train_sampler, batch_size):
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers_cpu, 
        pin_memory=pin_memory,
        drop_last=True
    )

    return train_loader


# In[21]:


def training_iteration(i, batch, scaler, model, optimizer, device, epoch_losses):
    # Move data to device
    inputs_nt = batch["nt_encodings"].to(device, non_blocking=True)
    inputs_aa = batch["aa_encodings"]["input_ids"].to(device, non_blocking=True)
    attention_mask_aa = batch["aa_encodings"]["attention_mask"].to(device, non_blocking=True)
    labels = batch['labels'].to(device, non_blocking=True).long()

    # Set all padding positions to a valid label (e.g., 0)
    labels = labels.clone()
    labels[labels == 9] = 0

    # Forward pass - use mixed precision if available
    if scaler is not None:
        with autocast("cuda"):
            outputs = model(inputs_nt, inputs_aa, attention_mask_aa, labels=labels)
            loss = outputs['loss']
    else:
        # Regular forward pass without mixed precision
        outputs = model(inputs_nt, inputs_aa, attention_mask_aa, labels=labels)
        loss = outputs['loss']

    # Backward pass - handle both mixed precision and regular training
    optimizer.zero_grad()

    if scaler is not None:
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        # Regular backward pass
        loss.backward()
        optimizer.step()

    # Store loss value and cleanup immediately
    epoch_losses.append(loss.item())

    #Periodic cleanup
    #if i % 20 == 0:
    #    clear_memory()

    #print weights in transition matrix
    if i % 10000 == 0:
        print(model.CRF.crf.transitions)

    # Immediate cleanup to prevent memory leaks
    del outputs, loss
    del inputs_nt, inputs_aa, attention_mask_aa, labels

    return epoch_losses


# In[22]:


def show_examples(counter, v_labels, padding_mask, predictions, seq_descs_batch):
    # Show hard examples as demonstration
    if counter == 0:
        for seq_i in range(min(32, v_labels.shape[0])):
            mask = ~padding_mask[seq_i]
            if seq_descs_batch[seq_i] not in ["non-coding", "coding", "coding_with_substitutions"]:
                print("Sequence type:", seq_descs_batch[seq_i])
                print("Labels:\n", v_labels[seq_i][mask].cpu().numpy().astype(float))
                print("Predictions:\n", predictions[seq_i][mask].cpu().numpy().astype(float))
                print("\n")


# In[23]:


def adjust_sequence_type_distribution_in_training(sequence_types, 
                                                  tracker, 
                                                  type_performance, 
                                                  type_weights,
                                                  weight_samplers_changed,
                                                  drop_patience,
                                                  min_weight):
    for desc in sequence_types:
        current_loss = tracker.get_metrics().get(desc, None)
        if current_loss is not None:
            history = type_performance[desc]['loss_history']
            history.append(current_loss)
            # Check if loss improved (use a small tolerance) -> increase patience when loss doesn't improve
            if len(history) > 1 and history[-1] > min(history[:-1]):
                type_performance[desc]['patience'] += 1
            else:
                type_performance[desc]['patience'] = 0
            # If patience exceeded, reduce weight (but not below min_weight)
            if type_performance[desc]['patience'] >= drop_patience:
                old_weight = type_weights[desc]
                type_weights[desc] = max(min_weight, old_weight * 0.8)  # Reduce weight by 20%
                print(f"Reduced weight for {desc}: {old_weight} -> {type_weights[desc]}")
                type_performance[desc]['patience'] = 0  # Reset patience
                print("New type weights:", type_weights)
                weight_samplers_changed = True

    return weight_samplers_changed, history, type_weights, type_performance


# In[24]:


def objective(trial, train_data, val_data, sequence_types, seq_type_desc_fracs):
    #Define hyperparameter ranges to sample from
    depths_transformer_encoder_blocks = [2, 4, 6, 8]
    attention_heads = [2, 4]
    dropout_rates_1 = [0.1, 0.2, 0.3] #Dropout rate applied after ESM-2 encoding
    dropout_rates_2 = [0.2, 0.3, 0.4, 0.5] #dropout rate applied in transformer encoder layers
    dropout_rates_3 = [0.0, 0.05]
    lr_esm2_rates = [1e-5, 8e-6, 5e-6, 1e-6]      
    lr_scratch_rates = [8e-5, 5e-5, 1e-5, 5e-6]  
    act_functions = ["relu", "gelu"]
    weight_decays = [0.001, 0.01, 0.1]
    crf_init_weights = [10]

    #Define trial suggestions; set hyperparameters
    depth_transformer_encoder_blocks = trial.suggest_categorical('depth_transformer_encoder_blocks', depths_transformer_encoder_blocks)
    n_attention_heads = trial.suggest_categorical('n_attention_heads', attention_heads)
    dropout_rate_1 = trial.suggest_categorical('dropout_rate_1', dropout_rates_1)
    dropout_rate_2 = trial.suggest_categorical('dropout_rate_2', dropout_rates_2)
    dropout_rate_3 = trial.suggest_categorical('dropout_rate_3', dropout_rates_3)
    lr_esm2 = trial.suggest_categorical('lr_esm2', lr_esm2_rates)
    lr_scratch = trial.suggest_categorical('lr_scratch', lr_scratch_rates)
    act_function = trial.suggest_categorical('act_function', act_functions)
    weight_decay = trial.suggest_categorical('weight_decay', weight_decays)
    crf_init_weight = trial.suggest_categorical('crf_init_weight', crf_init_weights)

    batch_size = 64

    print(f"depth_transformer_encoder_blocks: {depth_transformer_encoder_blocks}\n \
            n_attention_heads: {n_attention_heads}\n \
            dropout_rate_1: {dropout_rate_1}\n \
            dropout_rate_2: {dropout_rate_2}\n \
            dropout_rate_3: {dropout_rate_3}\n \
            lr_esm2: {lr_esm2}\n \
            lr_scratch: {lr_scratch}\n \
            act_function: {act_function}\n \
            weight_decay: {weight_decay}\n \
            crf_init_weight: {crf_init_weight}")

    wandb.init(project=wandb_project_name, 
                config={
                        "depth_transformer_encoder_blocks": depth_transformer_encoder_blocks,
                        "n_attention_heads": n_attention_heads,
                        "dropout_rate_1": dropout_rate_1,
                        "dropout_rate_2": dropout_rate_2,
                        "dropout_rate_3": dropout_rate_3,
                        "act_function": act_function,
                        "weight_decay": weight_decay,
                        "crf_init_weight": crf_init_weight,
                        "lr_esm2": lr_esm2,
                        "lr_scratch": lr_scratch},
                        name = f"trial_{trial.number}")

    type_weights = {st: 1.0 for st in sequence_types}  # Initial sampling weights
    min_weight = 0.5  # Minimum weight to keep (avoid complete removal)

    min_weight_sum = min_weight * len(sequence_types) # Minimum sum of weights to keep
    print(f"Minimum sum of weights before stopping: {min_weight_sum}")

    # Create initial weighted sampler
    train_sampler = create_weighted_sampler(train_data, type_weights)

    #Define data loaders
    train_loader = adjust_train_sample_distribution(train_data, train_sampler, batch_size)

    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers_cpu,  
        pin_memory=pin_memory)

    model = initialize_model(device,
                            num_layers = depth_transformer_encoder_blocks,
                            n_attention_heads = n_attention_heads,
                            dropout_rate_1=dropout_rate_1,
                            dropout_rate_2=dropout_rate_2,
                            dropout_rate_3=dropout_rate_3,
                            act_function = act_function,
                            crf_init_weight = crf_init_weight)

    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    #print(f"Model dimensions: {print_model_dimensions(model)}")

    #Define settings for training
    epochs = 30
    steps_per_epoch = len(train_data) / batch_size                               #The number of steps taken per epoch
    eval_per_epoch = 6 #MODIFY                                                          #The number of evaluations made per epoch
    steps_between_evals = int(steps_per_epoch / eval_per_epoch)                  #The number of steps to take between each evaluation

    #Initialize the loss tracker
    tracker = CategoricalLossTracker(sequence_types)

    #Define the optimizer
    optimizer = torch.optim.Adam([
        {'params': model.sequence_encoder.parameters(), 'lr': lr_esm2},
        {'params': model.TransformerEncoderBlock.parameters(), 'lr': lr_scratch},
        {'params': model.CRF.parameters(), 'lr': lr_scratch}], 
        weight_decay=weight_decay)

    #Initialize variables for early stopping
    best_val_loss = float('inf')
    threshold_patience = 6  #Number of evaluations with no improvement to wait before stopping #MODIFY
    counter_patience = 0

    #Initialize variables for training loop
    step = 0  # Global step counter
    val_times_counter = 0

    # Initialize mixed precision scaler if using CUDA
    scaler = GradScaler() if "cuda" in device_type else None
    if scaler is not None:
        print("Mixed precision training enabled")

    #Training loop
    for epoch in range(epochs):
        model.train()
        epoch_losses = []  # Reset at start of each epoch
        batches_processed = 0  # Track actual batches processed this epoch

        for i, batch in enumerate(train_loader):
            #Run training iteration
            epoch_losses = training_iteration(i, batch, scaler, model, optimizer, device, epoch_losses)
            batches_processed += 1

            #Validation step
            if step % steps_between_evals == 0 and step > 0:
                clear_memory()
                model.eval()
                val_losses = []
                all_val_true_sequences = []
                all_val_pred_sequences = []
                all_val_sequence_types = [] 

                with torch.no_grad():
                    for counter, val_batch in enumerate(val_loader):
                        v_inputs_nt = val_batch["nt_encodings"].to(device, non_blocking=True)
                        v_inputs_aa = val_batch["aa_encodings"]["input_ids"].to(device, non_blocking=True)
                        v_attention_aa = val_batch["aa_encodings"]["attention_mask"].to(device, non_blocking=True)
                        v_labels_original = val_batch['labels'].to(device, non_blocking=True).long()

                        # Save padding mask before overwriting v_labels
                        padding_mask = (v_labels_original == 9)
                        valid_mask = ~padding_mask  # This is what we'll use consistently

                        # Set all padding positions to a valid label (e.g., 0)
                        # Create modified labels for model forward pass
                        v_labels_modified = v_labels_original.clone()
                        v_labels_modified[padding_mask] = 0  # Set padding to valid label for forward pass


                        # Forward pass for validation
                        if scaler is not None:
                            with autocast("cuda"):
                                v_outputs = model(v_inputs_nt, v_inputs_aa, v_attention_aa, v_labels_modified)
                                v_loss = v_outputs['loss']
                        else:
                            v_outputs = model(v_inputs_nt, v_inputs_aa, v_attention_aa, v_labels_modified)
                            v_loss = v_outputs['loss']

                        val_losses.append(v_loss.item())

                        # Get predictions and process sequences immediately
                        predictions = model.CRF.crf.decode(v_outputs['logits'], mask=valid_mask)

                        # Pad predictions to match v_labels shape for downstream code
                        # efficient - create tensor directly and pad
                        batch_size, seq_len = v_labels_modified.shape
                        predictions_tensor = torch.full((batch_size, seq_len), 9, 
                                                        dtype=torch.long, device=v_labels_modified.device)

                        for i_pred, seq in enumerate(predictions):
                            seq_len_actual = len(seq)
                            if seq_len_actual > 0:
                                predictions_tensor[i_pred, :seq_len_actual] = torch.tensor(seq, 
                                                                                    dtype=torch.long, 
                                                                                    device=v_labels_modified.device)
                        predictions = predictions_tensor

                        # Process category-specific losses (keep your existing logic)
                        seq_descs_batch = val_batch['seq_desc']
                        seq_len = v_inputs_aa.size(1) - 2

                        for desc in tracker.categories:
                            # Find which sequences in the batch belong to this desc
                            desc_mask = torch.tensor([d == desc for d in seq_descs_batch], device=device)
                            if desc_mask.any():
                                # Select only the relevant sequences
                                desc_logits = v_outputs["logits"][desc_mask]           # [num_desc, seq_len, 4]

                                # IMPORTANT: Use the same label preprocessing as the main loss
                                desc_labels_original = v_labels_original[desc_mask]     # Original labels
                                desc_labels_modified = v_labels_modified[desc_mask]     # Modified labels (padding=0)
                                desc_valid_mask = valid_mask[desc_mask]                 # Valid positions mask

                                # Option 1: Use modified labels (consistent with forward pass)
                                desc_crf_loss = -model.CRF.crf(desc_logits, desc_labels_modified, 
                                                            mask=desc_valid_mask, reduction='mean')

                                tracker.update(desc, desc_crf_loss)

                                del desc_logits, desc_labels_original, desc_labels_modified, desc_valid_mask, desc_crf_loss

                            del desc_mask

                        # Process each sequence in batch immediately
                        for seq_idx in range(v_labels_original.shape[0]):
                            mask = ~padding_mask[seq_idx]  # Use NOT padding_mask to select valid positions
                            if mask.any():  # Skip empty sequences
                                true_seq = v_labels_original[seq_idx][mask].cpu().numpy()
                                pred_seq = predictions[seq_idx][mask].cpu().numpy()
                                seq_type = seq_descs_batch[seq_idx]  # Get sequence type

                                all_val_true_sequences.append(true_seq)
                                all_val_pred_sequences.append(pred_seq)
                                all_val_sequence_types.append(seq_type)

                        #Monitor hard examples as demonstration of how well classification is going
                        show_examples(counter, v_labels_original, padding_mask, predictions, seq_descs_batch)

                        # Clean up each validation batch immediately
                        del v_inputs_nt, v_inputs_aa, v_attention_aa, v_labels_original, v_outputs, v_loss
                        del predictions, padding_mask

                    clear_memory()

                # Calculate validation loss and other final metrics for validation loop
                if val_losses:
                    val_avg_loss = sum(val_losses) / len(val_losses)
                else:
                    val_avg_loss = float('inf')

                #Implement pruning strategy
                trial.report(val_avg_loss, step)

                #Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if all_val_true_sequences and all_val_pred_sequences:
                    sequence_metrics = calculate_sequence_accuracy_metrics(all_val_true_sequences, 
                                                                        all_val_pred_sequences,
                                                                        all_val_sequence_types)
                else:
                    sequence_metrics = {}

                # Calculate training loss - use recent batches only
                if epoch_losses:
                    train_avg_loss = sum(epoch_losses[-steps_between_evals:]) / min(len(epoch_losses), steps_between_evals)
                else:
                    train_avg_loss = 0.0

                # Early stopping check
                if val_avg_loss < best_val_loss:
                    best_val_loss = val_avg_loss
                    #torch.save(model.state_dict(), f"../../../data/processed_data/model_data/single_rf_crf/model_without_errors/models/{project_name}_trained.pth")
                    counter_patience = 0
                else:
                    counter_patience += 1

                if counter_patience >= threshold_patience:
                    print("Early stopping triggered!")
                    break

                #Log metrics
                val_times_counter = log_evaluation_metrics(epoch, train_avg_loss, val_avg_loss, best_val_loss, tracker, sequence_metrics, val_times_counter, sequence_types, seq_type_desc_fracs)

                #Clean up validation data
                del val_losses, all_val_true_sequences, all_val_pred_sequences
                clear_memory()

                model.train()  # Back to training mode

            # Increment global step counter after each batch
            step += 1

        # Check if early stopping was triggered during validation
        if counter_patience >= threshold_patience:
            break

    wandb.finish()
    return best_val_loss


# # Main code

# In[25]:


np.random.seed(42)

train_data, val_data, sequence_types, seq_type_desc_fracs = load_and_process_data(max_len)

#train_indices = np.random.choice(len(train_data), size=min(32*500, len(train_data)), replace=False)
#val_indices = np.random.choice(len(val_data), size=min(32*50, len(val_data)), replace=False)

#train_data = torch.utils.data.Subset(train_data, train_indices.tolist())
#val_data = torch.utils.data.Subset(val_data, val_indices.tolist())

#Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize',   #Minimize loss
                            pruner=optuna.pruners.HyperbandPruner())
study.optimize(
    lambda trial: objective(trial, train_data, val_data, sequence_types, seq_type_desc_fracs), 
    n_trials=30
)

model_config = {
    "depth_transformer_encoder_blocks": study.best_params['depth_transformer_encoder_blocks'],
    "n_attention_heads": study.best_params['n_attention_heads'],
    "dropout_rate_1": study.best_params['dropout_rate_1'],
    "dropout_rate_2": study.best_params['dropout_rate_2'],
    "dropout_rate_3": study.best_params['dropout_rate_3'],
    "lr_esm2": study.best_params['lr_esm2'],
    "lr_scratch": study.best_params['lr_scratch'],
    "act_function": study.best_params['act_function'],
    "weight_decay": study.best_params['weight_decay'],
    "crf_init_weight": study.best_params['crf_init_weight']}

os.makedirs("../../../data/processed_data/model_data/single_rf_crf/model_with_errors/hyperparameter_configs/", exist_ok = True)

#Save the dictionary as a JSON file
file_path = f'../../../data/processed_data/model_data/single_rf_crf/model_with_errors/hyperparameter_configs/esm2_8m_nt_hyperparameters.json'
with open(file_path, 'w') as json_file:
    json.dump(model_config, json_file, indent=4)

print(f"Model configuration saved to {file_path}.")

