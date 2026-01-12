import numpy as np
import pandas as pd
import optuna
import gc
import math
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef
from transformers import AutoTokenizer, AutoModel
from torch.amp import GradScaler, autocast

import wandb
import os
import json
import pickle

from torchcrf import CRF

torch.cuda.empty_cache() #Clear the GPU memory cache
pd.options.mode.chained_assignment = None

#Configure CUDA memory allocations (manage fragmentation in the GPU memory)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device_type = device.type  # "cuda", "mps", or "cpu"

if device_type == "mps":
    input_data_dir_path = "../../../data/processed_data/model_data/shared_crf/model_with_errors"
    num_workers_cpu = 0
    pin_memory = False
elif device_type == "cuda":
    input_data_dir_path = "/tmp/nrt204/FragmentPredictor/data/processed_data/model_data/shared_crf/model_with_errors" #TEST ON SCARB CLUSTER
    num_workers_cpu = 4
    pin_memory = True

print("Device type:", device_type, flush = True)

#Make sure dir to store model exists
os.makedirs(f"{input_data_dir_path}/models_optuna_codon_encoding/", exist_ok=True)
os.makedirs(f"{input_data_dir_path}/hyperparameter_configs/", exist_ok=True)

#dir in wandb to place runs
wandb_project_name = "tune_shared_codon_encoding_updated" #modify

max_len = 100

######################################################################################################################################################################################################
############################################################################################Define functions##########################################################################################
######################################################################################################################################################################################################

def set_seed(seed):
    """
    Set seed for reproducibility

    Args:
        seed (int): seed value to set
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clear_memory():
    """ 
    Clear GPU and CPU memory caches to free up resources.
    Empties CUDA cache if GPU is available and triggers garbage collection.
    """
    #Memory clean up function
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

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


def process_nt_sequences_to_codons(nt_sequences, max_aa_len):
    """
    Convert nucleotide sequences from dimension (4, nucleotide_seq_len) to dimension (12, nucleotide_seq_len/3) format
    by grouping every 3 nucleotides (1 codon) together.

    Args:
        nt_sequences: List of tensors with shape (4, nucleotide_seq_len)

    Returns:
        List of tensors with shape (12, nucleotide_seq_len/3)
    """
    processed_sequences = []

    for seq in nt_sequences:
        #seq has shape (4, nucleotide_seq_len) -> reshape to (4, nucleotide_seq_len/3, 3) to group every 3 nucleotides
        seq_reshaped = seq.view(4, max_aa_len, 3)

        #Transpose to (nucleotide_seq_len/3, 4, 3) then reshape to (nucleotide_seq_len/3, 12)
        seq_transposed = seq_reshaped.transpose(0, 1)  # (nucleotide_seq_len/3, 4, 3)
        seq_formatted = seq_transposed.reshape(max_aa_len, 12)  # (nucleotide_seq_len/3, 12)

        processed_sequences.append(seq_formatted)

    return processed_sequences


class SeqDataset(torch.utils.data.Dataset):
    """
    Dataset class for multi-reading-frame sequence data.
    
    Stores nucleotide encodings, amino acid encodings, and labels for all three
    reading frames (rf0, rf1, rf2) along with sequence descriptions.
    
    Args:
        nt_encodings_rf0 (list): List of nucleotide codon encodings (max_aa_len, 12) for reading frame 0
        labels_rf0 (list): List of numpy arrays (int8) with per-position labels for reading frame 0
        nt_encodings_rf1 (list): List of nucleotide codon encodings (max_aa_len, 12) for reading frame 1
        labels_rf1 (list): List of numpy arrays (int8) with per-position labels for reading frame 1
        nt_encodings_rf2 (list): List of nucleotide codon encodings (max_aa_len, 12) for reading frame 2
        labels_rf2 (list): List of numpy arrays (int8) with per-position labels for reading frame 2
        label_encodings (np.ndarray): Padded array of shape (num_samples, max_len-2) with mapped label sequences (int8)
        seq_desc (list): List of sequence description strings/identifiers
        
    Returns:
        Dictionary containing all encodings and labels for a single sequence across
        all three reading frames, with tensors converted to appropriate dtypes.
    """
    def __init__(self, nt_encodings_rf0, labels_rf0, 
                 nt_encodings_rf1, labels_rf1, 
                 nt_encodings_rf2, labels_rf2, 
                 label_encodings,
                 seq_desc):

        self.nt_encodings_rf0 = nt_encodings_rf0
        self.labels_rf0 = labels_rf0  

        self.nt_encodings_rf1 = nt_encodings_rf1
        self.labels_rf1 = labels_rf1  

        self.nt_encodings_rf2 = nt_encodings_rf2
        self.labels_rf2 = labels_rf2  

        self.label_encodings = label_encodings
        self.seq_desc = seq_desc

    def __getitem__(self, idx):
        item = {
            'nt_encodings_rf0': torch.as_tensor(self.nt_encodings_rf0[idx], dtype=torch.float32),
            'labels_rf0': torch.from_numpy(self.labels_rf0[idx]) if isinstance(self.labels_rf0[idx], np.ndarray) else torch.tensor(self.labels_rf0[idx], dtype=torch.float32),

            'nt_encodings_rf1': torch.as_tensor(self.nt_encodings_rf1[idx], dtype=torch.float32),
            'labels_rf1': torch.from_numpy(self.labels_rf1[idx]) if isinstance(self.labels_rf1[idx], np.ndarray) else torch.tensor(self.labels_rf1[idx], dtype=torch.float32),

            'nt_encodings_rf2': torch.as_tensor(self.nt_encodings_rf2[idx], dtype=torch.float32),
            'labels_rf2': torch.from_numpy(self.labels_rf2[idx]) if isinstance(self.labels_rf2[idx], np.ndarray) else torch.tensor(self.labels_rf2[idx], dtype=torch.float32),

            'label_encodings': torch.from_numpy(self.label_encodings[idx]) if isinstance(self.label_encodings[idx], np.ndarray) else torch.tensor(self.label_encodings[idx], dtype=torch.float32),
            'seq_desc': self.seq_desc[idx]
        }
        return item

    def __len__(self):
        return len(self.label_encodings)


def encode_data(processed_samples_df, max_len):
    """ 
    Encode data samples to fit model input format. 

    Args:
        processed_samples_df (dataframe): Dataframe with input dataset.
        max_len (int): Max ESM model input length.

    Returns:
        - dataset (dict): nested dictionary with data formatted to fit model input.
        - list of sequence types. 
    """

    #Initialize
    encodings_nt = {}
    labels = {}
    max_nt_len = max_len * 3

    #Label processing; shared label sequence (mapped from rf0, rf1, rf2 labels)
    if isinstance(processed_samples_df["label_encodings"].iloc[0], str):
        processed_samples_df["label_encodings"] = processed_samples_df["label_encodings"].apply(eval)

    #Convert overall (shared) labels to arrays and pad to max length
    label_arrays = [np.array(x, dtype=np.int8) for x in processed_samples_df["label_encodings"]]
    padded_labels = np.full((len(label_arrays), max_len), -1, dtype=np.int8)

    for i, arr in enumerate(label_arrays):
        length = min(len(arr), max_len)
        padded_labels[i, :length] = arr[:length]

    #Proces data from each RF separately 
    for rf in ["rf0", "rf1", "rf2"]:

        #====Label processing====#
        if isinstance(processed_samples_df[f"{rf}_labels"].iloc[0], str):
            processed_samples_df[f"{rf}_labels"] = processed_samples_df[f"{rf}_labels"].apply(eval)

        #Convert to numpy arrays more efficiently
        label_arrays = [np.array(x, dtype=np.int8) for x in processed_samples_df[f"{rf}_labels"]]

        labels[rf] = label_arrays

        #====Nucleotide sequence processing====#
        # Pad the sequences
        processed_samples_df[f"{rf}_seq_nt"] = processed_samples_df[f"{rf}_seq_nt"].apply(
            lambda seq: seq + 'N' * (max_nt_len - len(seq)) if len(seq) < max_nt_len else seq)

        nt_sequences = [one_hot_encode(seq) for seq in processed_samples_df[f"{rf}_seq_nt"]] #[max_nt_len, 4, num_seqs]

        # Process nt_sequences to codon-based format (3*4=12, max_nt_len/3)
        nt_encodings_rf = process_nt_sequences_to_codons(nt_sequences, max_len)
        encodings_nt[rf] = nt_encodings_rf

    seq_descriptions = processed_samples_df["seq_desc"].tolist()

    dataset = SeqDataset(encodings_nt["rf0"], labels["rf0"],
                         encodings_nt["rf1"], labels["rf1"],
                         encodings_nt["rf2"], labels["rf2"],
                         padded_labels, seq_descriptions)

    return dataset, list(set(seq_descriptions))


def load_and_process_data(max_len):
    """
    Main function that loads and processes all data efficiently.

    Args: 
        max_len: The model input length (amino acid/codon length + 2 for special tokens)

    Returns: 
        train_data: The encoded training dataset 
        val_data: The subsampled and encoded validation dataset
        sequence_types (list): list of all sequence types in dataset
        seq_type_desc_fracs: dict with distribution of sequence types in training data set
    """
    #Load data
    train_set = pd.read_csv(
        f"{input_data_dir_path}/datasets_model/train_100_genomes.csv.gz", #Hyperparameter tuning on the 100 genome dataset 
        index_col=None, 
        compression="gzip")
    
    val_set = pd.read_csv(
        f"{input_data_dir_path}/datasets_model/val.csv.gz", 
        index_col=None, 
        compression="gzip")
    
    seq_type_desc_fracs = (val_set['seq_desc'].value_counts(normalize=True)).to_dict()
    
    #Create a combined stratification label for accession and sequence type
    val_set["accession_seq_desc_merged"] = val_set["accession"].astype(str) + "_" + val_set["seq_desc"].astype(str) 

    ##Validate on 115k samples = 5 % of val set (0.05) following the original distribution stratified on accession and sequence type
    val_set = (val_set.groupby("accession_seq_desc_merged", group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=42))) #0.05 #MODIFY

    #Create a combined stratification label for accession and sequence type
    #train_set["accession_seq_desc_merged"] = train_set["accession"].astype(str) + "_" + train_set["seq_desc"].astype(str) #DELETE

    ##Validate on 115k samples = 5 % of val set (0.05) following the original distribution stratified on accession and sequence type
    #train_set = (train_set.groupby("accession_seq_desc_merged", group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=42))) ##DELETE

    print("Training data samples : ", train_set.shape[0])
    print("Validation data samples during training: ", val_set.shape[0])
    print("Distribution of sequence types in training set:", seq_type_desc_fracs)

    #Process training data
    train_data, sequence_types = encode_data(train_set, max_len)

    #Process validation data
    val_data, _ = encode_data(val_set, max_len)

    return train_data, val_data, sequence_types, seq_type_desc_fracs


def load_full_validation_set(max_len):
    """ 
    Load full validation set for final validation. 

    Args:
        max_len: The model input length (amino acid/codon length + 2 for special tokens)

    Returns: 
        val_data: The full encoded validation dataset
    """
    # Load data
    val_set = pd.read_csv(
        f"{input_data_dir_path}/datasets_model/val.csv.gz",
        index_col=None, 
        compression="gzip")

    print("Validation data samples in full set: ", val_set.shape[0])

    #Create a combined stratification label for accession and sequence type
    #val_set["accession_seq_desc_merged"] = val_set["accession"].astype(str) + "_" + val_set["seq_desc"].astype(str) 

    ##Validate on 115k samples = 5 % of val set (0.05) following the original distribution stratified on accession and sequence type
    #val_set = (val_set.groupby("accession_seq_desc_merged", group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))) #0.05 #MODIFY


    #Process validation data
    val_data, _ = encode_data(val_set, max_len)

    #Create model loader for the full validation set
    val_loader = DataLoader(
        val_data, 
        batch_size=350, 
        shuffle=False, 
        num_workers=num_workers_cpu,  
        pin_memory=pin_memory)

    return val_loader


def create_codon_padding_mask(frame_sequences):
    """
    Create padding mask for codon sequences.

    Args:
        frame_sequences: Tensor of shape (batch, 100, 12)
                        One-hot encoded codons

    Returns:
        mask: Boolean tensor (batch, 100)
              True = padding position (ignore in attention)
              False = real codon (use in attention)
    """
    # Sum across the 12 dimensions - padded positions will be all zeros
    mask = (frame_sequences.sum(dim=-1) == 0)
    return mask


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        # Pre-compute for efficiency, but can extend dynamically
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)

        if seq_len > self.pe.size(0):
            # Dynamically compute for longer sequences
            pe = self._compute_pe(seq_len, self.d_model, x.device)
            return x + pe[:seq_len, :]

        return x + self.pe[:seq_len, :]

    @staticmethod
    def _compute_pe(seq_len, d_model, device):
        pe = torch.zeros(seq_len, d_model, device=device)
        position = torch.arange(0, seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


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
                 act_function,
                 num_labels):
        super().__init__()

        hidden_size_merged = hidden_size

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


    def forward(self, encoded_seqs_nt, attention_mask):
        """
        Forward pass through the Transformer encoder block and linear classifier.

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, hidden_size]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]

        Returns:
            torch.Tensor of logits [batch_size, seq_len, num_labels]
        """

        # Original transformer processing
        codon_embeddings = encoded_seqs_nt.permute(1, 0, 2)  # [seq_len, batch, hidden]
        attention_mask_transformer = ~attention_mask.bool()

        #Pass through transformer encoder layers
        codon_embeddings = self.encoder(codon_embeddings, src_key_padding_mask=attention_mask_transformer)

        codon_embeddings = codon_embeddings.permute(1, 0, 2)  # [batch, seq_len, hidden]

        #Layer normalization
        codon_embeddings = self.norm(codon_embeddings)

        #Get RF-specific class logits out
        logits = self.linear(codon_embeddings)  # [batch, seq_len, num_labels]

        return logits

class LinearChainCRF(nn.Module):
    """
    Neural network module that applies a Conditional Random Field (CRF) layer for structured prediction.
    Designed to handle dynamic reading-frame (RF) combination labels and enforce biologically 
    constrained transitions between RF states.

    Args:
        mapping_dict_to_class (dict): Mapping from integer label indices to tuples representing the corresponding reading-frame combination (e.g., `{0: (0, 1, 2), 1: (1, 3, 5), ...}`).
        num_class_labels (int, optional): Total number of class labels. If not provided, it is inferred from `mapping_dict_to_class`.

    Attributes:
        shared_rf_labels_mapping (dict): Stores the mapping from label indices to RF combinations.
        crf (torchcrf.CRF): Linear-chain Conditional Random Field layer that models label dependencies across the sequence.
        legal_transitions (dict): Defines valid transitions between RF states for each of the three reading frames.
        biologically_valid_mask (torch.BoolTensor): Mask indicating which transitions are allowed (True) or constrained (False).
        frequent_transition_mask (torch.BoolTensor): Mask indicating which transitions are frequent self-transitions.
    """

    def __init__(self, 
                 mapping_dict_to_class, 
                 transition_weight,
                 num_class_labels=None):
        super().__init__()

        #Load shared class mapping
        self.shared_rf_labels_mapping = mapping_dict_to_class

        #Determine number of classes if not provided
        if num_class_labels is None:
            num_class_labels = len(self.shared_rf_labels_mapping)

        self.crf = CRF(num_tags=num_class_labels, batch_first=True)

        #Define allowed RF transition rules
        self.legal_transitions = {
            0: {0, 2, 4},
            1: {1, 3, 5},
            2: {1, 5},
            3: {0, 2, 4},
            4: {1, 3, 5},
            5: {0, 2, 4}
        }

        self.biologically_valid_mask = torch.ones_like(self.crf.transitions, dtype=torch.bool)
        self.frequent_transition_mask = torch.zeros_like(self.crf.transitions, dtype=torch.bool)
        
        self._create_biologically_valid_mask()
        self._create_frequent_transition_mask()

        #Initialize transitions with three-tier scheme
        with torch.no_grad():
            # Stage 1: All illegal transitions → -10 (forbidden)
            self.crf.transitions[~self.biologically_valid_mask] = -10
            
            # Stage 2: Legal but infrequent transitions → transition_weight + small noise (discouraged but diverse)
            legal_infrequent = self.biologically_valid_mask & ~self.frequent_transition_mask
            noise = torch.randn_like(self.crf.transitions[legal_infrequent]) * 0.1
            self.crf.transitions[legal_infrequent] = transition_weight + noise
            
            # Stage 3: Frequent self-transitions → 0 (neutral/encouraged)
            self.crf.transitions[self.frequent_transition_mask] = 0 

    def _is_legal_transition(self, from_rf, to_rf):
        """Check if transition from one RF combination to another is legal"""
        from_rf0, from_rf1, from_rf2 = from_rf
        to_rf0, to_rf1, to_rf2 = to_rf

        # All three RFs must have legal transitions
        rf0_legal = to_rf0 in self.legal_transitions[from_rf0]
        rf1_legal = to_rf1 in self.legal_transitions[from_rf1]
        rf2_legal = to_rf2 in self.legal_transitions[from_rf2]

        return rf0_legal and rf1_legal and rf2_legal

    def _create_biologically_valid_mask(self):
        """
        Create a mask indicating which transitions should be constrained.
        False = constrained (will be set to penalty), True = learnable
        """
        num_labels = len(self.shared_rf_labels_mapping)
        legal_count = 0
        illegal_count = 0

        # Check all possible transitions between the labels
        for from_label in range(num_labels):
            for to_label in range(num_labels):
                from_rf = self.shared_rf_labels_mapping[from_label]
                to_rf = self.shared_rf_labels_mapping[to_label]

                if self._is_legal_transition(from_rf, to_rf):
                    # Legal transition - keep as learnable (True)
                    self.biologically_valid_mask[from_label, to_label] = True
                    legal_count += 1
                else:
                    # Illegal transition - mark as constrained (False)
                    self.biologically_valid_mask[from_label, to_label] = False
                    illegal_count += 1

        print(f"Legal transitions: {legal_count}")
        print(f"Illegal transitions: {illegal_count}")
        print(f"Total transitions: {legal_count + illegal_count}")
        print(f"Percentage legal: {legal_count/(legal_count + illegal_count)*100:.1f}%")

    def _create_frequent_transition_mask(self):
        """
        Create a mask for frequent self-transitions.
        These are transitions from a label to itself for the special RF combinations:
        (0,0,0)->(0,0,0), (1,0,0)->(1,0,0), (0,1,0)->(0,1,0), (0,0,1)->(0,0,1)
        """
        # Define the frequent RF combinations
        frequent_rf_combinations = {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)}
        
        # Find label indices corresponding to these RF combinations
        frequent_labels = []
        for label_idx, rf_combo in self.shared_rf_labels_mapping.items():
            if rf_combo in frequent_rf_combinations:
                frequent_labels.append(label_idx)
        
        # Mark self-transitions for these labels
        frequent_count = 0
        for label in frequent_labels:
            self.frequent_transition_mask[label, label] = True
            frequent_count += 1
        
        print(f"Frequent self-transitions: {frequent_count}")
        print(f"Frequent transition labels: {frequent_labels}")

    def forward(self, logits, attention_mask, labels=None):
        """
        Forward pass with CRF layer.

        Args: 
            logits (torch.Tensor): Emission scores of shape (batch_size, seq_len, num_labels).
            attention_mask (torch.Tensor): Mask where 1 indicates valid tokens.
            labels (torch.Tensor, optional): Gold label indices for training.
            sequence_class (torch.Tensor, optional): Sequence-level class indices for weighted loss.

        Returns: 
            dict: 
            During training (`labels` provided):
                    - **'loss' (torch.Tensor)**: Mean CRF loss over the batch.
                    - **'logits' (torch.Tensor)**: Input logits passed through the CRF.
                During inference (`labels` omitted):
                    - **'predictions' (list[list[int]])**: Decoded most probable label sequence per sample.
                    - **'logits' (torch.Tensor)**: Input logits passed through the CRF.
        """
        #Training
        if labels is not None:
            crf_mask = attention_mask.bool()
            #calculate log likelihood of the true label sequence under the CRF model for each sequence in batch (no reduction across batch)
            log_likelihood = self.crf(logits, labels, mask=crf_mask, reduction='none') #returns the log-likelihood value per true label sequence 

            #compute negative log likelihood loss averaged across batch
            loss = -log_likelihood.mean()

            return {'loss': loss, 'logits': logits}

        #Inference mode: decode the most probable label sequence using the Viterbi algorithm
        else:
            crf_mask = attention_mask.bool()
            predictions = self.crf.decode(logits, mask=crf_mask)
            return {'predictions': predictions, 'logits': logits}


class CDSPredictor(nn.Module):
    """
    Full model for CDS prediction combining transformer encoders per reading frame (RF0–RF2) with a shared CRF layer for structured, frame-consistent predictions.

    Args:
        esm2_model (nn.Module): Pretrained ESM-2 model used to extract amino acid embeddings.
        num_layers (int): Number of Transformer encoder layers per reading frame.
        n_attention_heads (int): Number of attention heads in each Transformer layer.
        d_model (int): dimension of FF layer to project input to and for use in FF layers in transformer encoders 
        dropout_rate_1 (float): Dropout rate applied in the sequence encoder.
        dropout_rate_2 (float): Dropout rate applied within the Transformer encoder layers.
        act_function (str or Callable): Activation function used in Transformer feedforward layers.
        num_encoded_labels (int): Number of combined label states used by the CRF.
        encoded_labels_mapping (dict): Mapping from integer label indices to RF combination tuples.

    Attributes:
        sequence_encoder (SequenceEncoder): Module that extracts amino acid embeddings from the pretrained ESM-2 model.
        TransformerEncoderBlock (TransformerEncoderBlock): Transformer encoder applied independently to each reading frame.
        linear_transform (nn.Linear): Linear projection layer mapping concatenated RF outputs (3*num_labels) to num_encoded_labels.
        CRF (LinearChainCRF): Conditional Random Field layer enforcing structured transitions between predicted RF combinations.
    """
    def __init__(self,
                 num_layers,
                 n_attention_heads,
                 d_model,
                 dropout_rate_1,
                 dropout_rate_2,
                 act_function,
                 transition_weight,
                 num_encoded_labels,
                 encoded_labels_mapping
                 ): 
        super(CDSPredictor, self).__init__()

        self.input_proj = nn.Linear(12, d_model)

        self.dropout_rate_1 = nn.Dropout(dropout_rate_1)

        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=500)

        self.TransformerEncoderBlock = TransformerEncoderBlock(
            hidden_size=d_model,
            num_layers=num_layers,
            n_attention_heads=n_attention_heads,
            dropout_rate_encoder=dropout_rate_2,
            act_function=act_function,
            num_labels=6)

        ##Linear layer to go from 3*C -> num_encoded_labels
        self.output_proj = nn.Linear(3*6, num_encoded_labels)

        self.CRF = LinearChainCRF(mapping_dict_to_class = encoded_labels_mapping,
                                  transition_weight = transition_weight,
                                  num_class_labels=num_encoded_labels)


    def forward(self, encoded_seqs_nt_rf0, 
                      encoded_seqs_nt_rf1,
                      encoded_seqs_nt_rf2,
                      labels=None):
        """
        Forward pass through the model with CRF support.

        Args:
            encoded_seqs_nt_rf0 (torch.Tensor): Encoded nucleotide sequences for RF0 [batch_size, seq_len, 12]
            encoded_seqs_nt_rf1 (torch.Tensor): Encoded nucleotide sequences for RF1 [batch_size, seq_len, 12]
            encoded_seqs_nt_rf2 (torch.Tensor): Encoded nucleotide sequences for RF2 [batch_size, seq_len, 12]
            labels (torch.Tensor, optional): True labels for CRF training [batch_size, seq_len]

        """

        #Create padding mask 
        padding_mask = create_codon_padding_mask(encoded_seqs_nt_rf0)

        #Project input vectors to d_model size
        encoded_embeddings_rf0 = self.dropout_rate_1(self.input_proj(encoded_seqs_nt_rf0))
        encoded_embeddings_rf1 = self.dropout_rate_1(self.input_proj(encoded_seqs_nt_rf1))
        encoded_embeddings_rf2 = self.dropout_rate_1(self.input_proj(encoded_seqs_nt_rf2))

        #Add positional encoding
        encoded_embeddings_rf0 = encoded_embeddings_rf0 + self.pos_encoding(encoded_embeddings_rf0)
        encoded_embeddings_rf1 = encoded_embeddings_rf1 + self.pos_encoding(encoded_embeddings_rf1)
        encoded_embeddings_rf2 = encoded_embeddings_rf2 + self.pos_encoding(encoded_embeddings_rf2)

        #Run through transformer encoder layers
        logits_rf0 = self.TransformerEncoderBlock(encoded_seqs_nt = encoded_embeddings_rf0, attention_mask=padding_mask)
        logits_rf1 = self.TransformerEncoderBlock(encoded_seqs_nt = encoded_embeddings_rf1, attention_mask=padding_mask)
        logits_rf2 = self.TransformerEncoderBlock(encoded_seqs_nt = encoded_embeddings_rf2, attention_mask=padding_mask) #output: [100, C]

        #Concatenate embeddings from each window
        codon_embeddings = torch.cat([logits_rf0, logits_rf1, logits_rf2], dim=-1) #output: [100, 3*C]

        #Project to shared label space
        logits_encoded_labels = self.output_proj(codon_embeddings) #output: [100, num_encoded_labels]

        crf_mask = ~padding_mask  # Invert: True = valid, False = padding

        output = self.CRF(
            logits=logits_encoded_labels,
            attention_mask=crf_mask, #Input any trimmed attention mask; applies to same positions as before
            labels=labels)

        return output


def print_model_dimensions(model):
    """Print model dimensions if needed."""
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

def count_parameters(model):
    """Print number of model parameters; both learnable and total"""
    total_params_learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total parameters: {total_params:,}")
    print(f"Total trainable parameters: {total_params_learnable:,}")


def initialize_model(device, num_layers, n_attention_heads, d_model, dropout_rate_1, dropout_rate_2, act_function, transition_weight):
    """
    Initialize the model and move it to the specified device.
    
    Args:
        device_type (str): The device to use for computation ("cuda", "mps", or "cpu").
        num_layers (int): Number of Transformer encoder layers per reading frame.
        n_attention_heads (int): Number of attention heads in each Transformer encoder layer.
        d_model (int): Dimension of FF-layers to project input encoding and to use in FF layers of sequence encoder blocks.
        dropout_rate_1 (float): Dropout rate applied in the sequence encoder.
        dropout_rate_2 (float): Dropout rate applied within the Transformer encoder layers.
        act_function (str or Callable): Activation function used in Transformer feedforward layers.


    Returns:
        model (nn.Module): The initialized model.
        mapping_dict_to_class (dict): dictionary mapping the shared label encodings to 3D rf-specific label combinations
    """

    print("Running on: ", device, flush = True)

    with open(f'{input_data_dir_path}/label_mappings/mapping_to_3d_vector.pkl', "rb") as mapping_file:
        mapping_dict_to_class = pickle.load(mapping_file)

    num_encoded_labels = len(mapping_dict_to_class.keys())
    print(f"Number of encoded label classes: {num_encoded_labels}")

    model = CDSPredictor(num_layers = num_layers,
                         n_attention_heads = n_attention_heads,
                         d_model=d_model,
                         dropout_rate_1 = dropout_rate_1, 
                         dropout_rate_2 = dropout_rate_2,
                         act_function = act_function,
                         transition_weight = transition_weight,
                         num_encoded_labels = num_encoded_labels,
                         encoded_labels_mapping = mapping_dict_to_class)
    model.to(device)

    if device.type == "cuda":
        print(f"Memory Allocated after loading model: {torch.cuda.memory_allocated(device) / 1024**3} GB")

    count_parameters(model)
    #print_model_dimensions(model)

    return model, mapping_dict_to_class



def calculate_sequence_accuracy_metrics(true_labels_list, predictions_list, sequence_types_list=None):
    """
    Calculate sequence-level accuracy metrics including:
        - MCC for overall and for each sequence type specifically.
        - Sequence type accuracy metrics

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

    total_sequences = len(true_labels_list)

    #Initalize
    perfect_sequences = 0
    high_accuracy_sequences = 0
    sequence_accuracies = []
    batch_size = 1000 #Process sequences in batches to save memory
    all_true_labels = []
    all_predictions = []

    #For sequence type-specific metrics
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

            #Add batch to overall arrays
            batch_all_true.extend(true_labels.tolist())
            batch_all_pred.extend(predictions.tolist())

            #Calculate accuracy
            accuracy = (true_labels == predictions).mean()
            batch_accuracies.append(accuracy)

            if accuracy == 1.0:
                perfect_sequences += 1
            if accuracy > 0.9:
                high_accuracy_sequences += 1

            #Handle type-specific data
            if batch_types is not None:
                seq_type = batch_types[idx]
                type_data = type_specific_data[seq_type]

                #Add to type-specific arrays
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

        #Clear batch data
        del batch_true, batch_pred, batch_accuracies, batch_all_true, batch_all_pred
        if batch_types is not None:
            del batch_types

    #Convert to numpy arrays
    all_true_labels = np.array(all_true_labels)
    all_predictions = np.array(all_predictions)

    # alculate overall MCC efficiently
    try:
        overall_mcc = matthews_corrcoef(all_true_labels, all_predictions) #Multi-class MCC
        if np.isnan(overall_mcc):
            overall_mcc = 0.0
    except:
        overall_mcc = 0.0


    #Build results
    results = {
        'fraction_perfect_sequences': perfect_sequences / total_sequences,
        'fraction_high_accuracy_sequences': high_accuracy_sequences / total_sequences,
        'overall_mcc': overall_mcc,
        'accuracy': np.mean(sequence_accuracies)
    }

    #Calculate type-specific metrics
    if sequence_types_list is not None:
        type_metrics = {}
        for seq_type, type_data in type_specific_data.items():
            if type_data['total_sequences'] > 0:
                #Convert type-specific data to numpy arrays
                type_true = np.array(type_data['true_labels'])
                type_pred = np.array(type_data['predictions'])

                #Calculate type-specific MCC
                try:
                    type_mcc = matthews_corrcoef(type_true, type_pred) #Multi-class MCC
                    if np.isnan(type_mcc):
                        type_mcc = 0.0
                except:
                    type_mcc = 0.0

                #Calculate type-specific metrics
                type_metrics[f'{seq_type}_mcc'] = type_mcc
                type_metrics[f'{seq_type}_accuracy'] = np.mean(type_data['sequence_accuracies'])
                type_metrics[f'{seq_type}_fraction_perfect'] = type_data['perfect_sequences'] / type_data['total_sequences']
                type_metrics[f'{seq_type}_fraction_high_accuracy'] = type_data['high_accuracy_sequences'] / type_data['total_sequences']

        #Add type-specific metrics to results
        results.update(type_metrics)

    return results


class CategoricalLossTracker:
    """ 
    A class to track losses for different sequence types during training.
    """
    def __init__(self, categories):
        self.categories = categories
        self.losses = {cat: [] for cat in categories} #Create empty list for each category

    def update(self, category, loss):
        #Extract average loss for the category in a batch
        self.losses[category].append(loss.item())

    def get_metrics(self):
        #Calculate the mean loss for each category
        return {cat: np.mean(losses) for cat, losses in self.losses.items()}


def create_weighted_sampler(dataset, sequence_types):
    """
    Create a weighted sampler based on sequence types and their weights (only used for initializing atm).
    """
    type_weights = {st: 1.0 for st in sequence_types}  #Initial sampling weights

    #Get sequence types for all samples in dataset
    seq_types_all = [dataset[i]['seq_desc'] for i in range(len(dataset))]

    #Create sample weights based on type_weights
    sample_weights = [type_weights.get(seq_type, 1.0) for seq_type in seq_types_all]

    #Create WeightedRandomSampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )

    return sampler


def adjust_train_sample_distribution(train_data, train_sampler, batch_size):
    """
    Data sampler for adjusting trainining sample distribution if necessary (only used for initializing at the moment, not dynamically)

    Args:
        train_data (torch.utils.data.Dataset): Dataset containing training samples.
        train_sampler (torch.utils.data.Sampler): Sampler to control which samples are drawn and their probabilities.
        batch_size (int): Number of samples per batch.

    Returns:
        torch.utils.data.DataLoader: DataLoader that yields batches from train_data according to train_sampler and batch_size.
    """

    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers_cpu, 
        pin_memory=pin_memory,
        drop_last=True
    )

    return train_loader


def log_evaluation_metrics(epoch, train_avg_loss, val_avg_loss, best_val_loss, tracker, sequence_metrics, val_times_counter, sequence_types):
    """ 
    Print evaluation metrics for the current epoch and log them to Weights & Biases (wandb).

    Args:
        epoch (int): Current training epoch.
        train_avg_loss (float): Average training loss for this epoch.
        val_avg_loss (float): Average validation loss for this epoch.
        best_val_loss (float): Best validation loss seen so far.
        tracker (object): Object providing access to per-sequence-type validation loss metrics via tracker.get_metrics().
        sequence_metrics (dict): Dictionary containing overall and type-specific sequence metrics, such as MCC, accuracy, fraction of perfect/high-accuracy sequences.
        val_times_counter (int): Counter tracking how many validation runs have been performed.
        sequence_types (list[str]): List of sequence types (e.g., categories) to include in type-specific metrics.

    Returns: 
        int: Updated validation counter (val_times_counter + 1).
    """

    #Build the print statement with type-specific metrics
    print_parts = [
        f"---Evaluation {val_times_counter}---\n",
        f"Train Loss: {train_avg_loss:.4f}\t\t",
        f"Val Loss: {val_avg_loss:.4f}\t\t"
    ]

    #Add overall metrics
    print_parts.extend([
        f"Overall MCC: {sequence_metrics.get('overall_mcc', 0):.4f}\t\t",
        f"Overall Accuracy: {sequence_metrics.get('accuracy', 0):.4f}\n"
    ])

    #Add loss metrics for each sequence type
    for seq_type in sequence_types:
        loss_val = tracker.get_metrics().get(seq_type, 0)
        print_parts.append(f"Val Loss {seq_type}: {loss_val:.4f}\t\t")

    print_parts.append("\n")

    #Add type-specific MCC and accuracy metrics
    for seq_type in sequence_types:
        type_mcc = sequence_metrics.get(f'{seq_type}_mcc', 0)
        type_acc = sequence_metrics.get(f'{seq_type}_accuracy', 0)
        if type_mcc != 0 or type_acc != 0:  # Only show if we have data for this type
            print_parts.extend([
                f"MCC {seq_type}: {type_mcc:.4f}\t\t",
                f"Acc {seq_type}: {type_acc:.4f}\t\t"
            ])

    #Print all metrics
    print("".join(print_parts), flush=True)

    #Build wandb logging dictionary
    wandb_log = {
        "epoch": epoch + 1,
        "train_loss": train_avg_loss,
        "val_loss": val_avg_loss,

        #Overall sequence metrics
        "val_fraction_perfect_sequences": sequence_metrics.get('fraction_perfect_sequences', 0),
        "val_fraction_high_accuracy_sequences": sequence_metrics.get('fraction_high_accuracy_sequences', 0),
        "val_overall_mcc": sequence_metrics.get('overall_mcc', 0),
        "val_accuracy": sequence_metrics.get('accuracy', 0)}

    #Add loss metrics for each sequence type
    for seq_type in sequence_types:
        wandb_log[f"val_loss_{seq_type}"] = tracker.get_metrics().get(seq_type, 0)

    #Add type-specific MCC and accuracy metrics
    for seq_type in sequence_types:
        #MCC metrics
        type_mcc = sequence_metrics.get(f'{seq_type}_mcc', 0)
        if type_mcc != 0:  # Only log if we have data
            wandb_log[f"val_mcc_{seq_type}"] = type_mcc

        #Accuracy metrics
        type_acc = sequence_metrics.get(f'{seq_type}_accuracy', 0)
        if type_acc != 0:  # Only log if we have data
            wandb_log[f"val_accuracy_{seq_type}"] = type_acc

        #Perfect and high accuracy sequence fractions
        type_perfect = sequence_metrics.get(f'{seq_type}_fraction_perfect', 0)
        if type_perfect != 0:
            wandb_log[f"val_fraction_perfect_{seq_type}"] = type_perfect

        type_high_acc = sequence_metrics.get(f'{seq_type}_fraction_high_accuracy', 0)
        if type_high_acc != 0:
            wandb_log[f"val_fraction_high_accuracy_{seq_type}"] = type_high_acc

    #Log lowest validation loss obtained throughout entire training
    wandb_log[f"best_val_loss"] = best_val_loss

    #Log to wandb
    wandb.log(wandb_log)

    return val_times_counter + 1


def training_iteration(i, batch, scaler, model, optimizer, device, epoch_losses):
    """
    Perform a single training iteration on one batch, including forward and backward passes, 
    optional mixed precision, and loss accumulation.

    Args:
        i (int): Index of the current batch in the epoch (used for logging/printing).
        batch (dict): Dictionary containing batch data with keys:
            - "nt_encodings_rf0/1/2": nucleotide encodings per reading frame.
            - "label_encodings": target labels for the CRF layer (with -1 for padding positions).
            - "seq_desc": List of string sequence types (e.g., "coding", "non-coding", "transition_start").
        scaler (torch.cuda.amp.GradScaler or None): Mixed precision scaler; if None, regular precision is used.
        model (nn.Module): CDS prediction model with transformer and CRF layers.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        device (torch.device): Device for computation ("cuda", "cpu", etc.).
        epoch_losses (list): List storing training loss values per batch for the current epoch.
        seq_type_to_idx (dict): Mapping from sequence type strings to integer indices for weighted loss.

    Returns:
        list: Updated `epoch_losses` list with the current batch loss appended.
    """

    #Move data to device
    inputs_nt_rf0 = batch["nt_encodings_rf0"].to(device, non_blocking=True)
    inputs_nt_rf1 = batch["nt_encodings_rf1"].to(device, non_blocking=True)
    inputs_nt_rf2 = batch["nt_encodings_rf2"].to(device, non_blocking=True)
    encoded_labels = batch['label_encodings'].to(device, non_blocking=True).long()

    #Forward pass; use mixed precision if available
    if scaler is not None:
        with autocast("cuda"):
            outputs = model(inputs_nt_rf0,
                          inputs_nt_rf1,
                          inputs_nt_rf2,
                          labels=encoded_labels)
            loss = outputs['loss']
    else:
        #Regular forward pass without mixed precision
        outputs = model(inputs_nt_rf0,
                       inputs_nt_rf1,
                       inputs_nt_rf2,
                       labels=encoded_labels)
        loss = outputs['loss']

    #Backward pass; handle both mixed precision and regular training
    optimizer.zero_grad()

    if scaler is not None:
        #Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        #Regular backward pass
        loss.backward()
        optimizer.step()

    #Store loss value
    epoch_losses.append(loss.item())

    #Print transition matrix for checkpoint
    if i % 10000 == 0:
        print(model.CRF.crf.transitions)

    if i % 1000 == 0:
        #Cleanup every 1000th batch to prevent memory leaks
        del outputs, loss, inputs_nt_rf0, inputs_nt_rf1, inputs_nt_rf2, encoded_labels

    return epoch_losses


def show_examples(v_labels, padding_mask, logits, seq_descs_batch, mapping_dict_to_class, model, device, valid_mask):
    """
    Show hard examples as demonstration per validation loop

    Args:
        counter: batch counter
        v_labels: encoded labels
        padding_mask: padding mask
        logits: model logits (not predictions)
        seq_descs_batch: sequence descriptions
        mapping_dict_to_class: label mapping
        model: model object (to access CRF)
        device: device
        valid_mask: valid positions mask
    """

    num_to_show = min(32, v_labels.shape[0])  #Show examples
    list_seq_types = ["none"]

    for seq_i in range(num_to_show):
        mask = ~padding_mask[seq_i]

        if seq_descs_batch[seq_i] not in list_seq_types:
            print("Sequence type:", seq_descs_batch[seq_i])

            #Get labels for this sequence
            labels_masked = v_labels[seq_i][mask].cpu().numpy().astype(int)

            #Decode prediction for sequence
            seq_logits = logits[seq_i:seq_i+1]  # [1, seq_len, num_classes]
            seq_valid_mask = valid_mask[seq_i:seq_i+1]  # [1, seq_len]

            pred_decoded = model.CRF.crf.decode(seq_logits, mask=seq_valid_mask)
            preds_masked = torch.tensor(pred_decoded[0], dtype=torch.long, device=device).cpu().numpy().astype(int)

            #Convert labels to RF vectors
            labels_rf = [mapping_dict_to_class[label] for label in labels_masked]

            #Convert predictions to RF vectors
            preds_rf = [mapping_dict_to_class[pred] for pred in preds_masked]

            #Extract individual RF sequences
            labels_rf0 = [rf[0] for rf in labels_rf]
            labels_rf1 = [rf[1] for rf in labels_rf]
            labels_rf2 = [rf[2] for rf in labels_rf]
            preds_rf0 = [rf[0] for rf in preds_rf]
            preds_rf1 = [rf[1] for rf in preds_rf]
            preds_rf2 = [rf[2] for rf in preds_rf]

            print("Encoded labels and preds:")
            print(v_labels[seq_i][mask].cpu().numpy().astype(float))
            print(preds_masked.astype(float))
            print()
            print("Labels RF0:", labels_rf0)
            print("Labels RF1:", labels_rf1)
            print("Labels RF2:", labels_rf2)
            print()
            print("Predictions RF0:", preds_rf0)
            print("Predictions RF1:", preds_rf1)
            print("Predictions RF2:", preds_rf2)
            print("\n")

        if seq_i == 3:
            #Only show a few of "easy-to-classify" samples
            list_seq_types = ["non-coding", "coding", "coding_with_substitutions"]



def objective(trial, train_data, val_data, val_loader_full, sequence_types, seq_type_desc_fracs):
    #Define hyperparameter ranges to sample from
    depths_transformer_encoder_blocks = [2, 4, 6, 8, 10, 12]
    attention_heads = [2, 4, 8]
    d_model_sizes = [128, 256, 512, 1024, 2048]
    dropout_rates_1 = [0.1, 0.2, 0.3] 
    dropout_rates_2 = [0.2, 0.3, 0.4, 0.5] #dropout rate applied in transformer encoder layers
    act_functions = ["relu", "gelu"]


    #Define trial suggestions; set hyperparameters
    depth_transformer_encoder_blocks = trial.suggest_categorical('depth_transformer_encoder_blocks', depths_transformer_encoder_blocks)
    n_attention_heads = trial.suggest_categorical('n_attention_heads', attention_heads)
    d_model = trial.suggest_categorical('d_model', d_model_sizes)
    dropout_rate_1 = trial.suggest_categorical('dropout_rate_1', dropout_rates_1)
    dropout_rate_2 = trial.suggest_categorical('dropout_rate_2', dropout_rates_2)
    lr_scratch = trial.suggest_float('lr_scratch', 1e-6, 1e-3, log=True)
    act_function = trial.suggest_categorical('act_function', act_functions)
    transition_weight = trial.suggest_float('transition_weight', -4, -1) #from probability from exp(-3) -> exp(-0.25) 

    batch_size = 32

    print(f"depth_transformer_encoder_blocks: {depth_transformer_encoder_blocks}\n \
            n_attention_heads: {n_attention_heads}\n \
            d_model: {d_model} \n \
            dropout_rate_1: {dropout_rate_1}\n \
            dropout_rate_2: {dropout_rate_2}\n \
            lr_scratch: {lr_scratch}\n \
            act_function: {act_function}\n \
            transition_weight: {transition_weight}")

    wandb.init(project=wandb_project_name, 
                config={
                        "depth_transformer_encoder_blocks": depth_transformer_encoder_blocks,
                        "n_attention_heads": n_attention_heads,
                        "d_model": d_model,
                        "dropout_rate_1": dropout_rate_1,
                        "dropout_rate_2": dropout_rate_2,
                        "act_function": act_function,
                        "transition_weight": transition_weight,
                        "lr_scratch": lr_scratch},
                        name = f"trial_{trial.number}")

    #Create initial weighted sampler for training and get data loaders
    train_sampler = create_weighted_sampler(train_data, sequence_types)
    train_loader = adjust_train_sample_distribution(train_data, train_sampler, batch_size)
    val_loader = DataLoader(
        val_data, 
        batch_size=350, 
        shuffle=True, 
        num_workers=num_workers_cpu,  
        pin_memory=pin_memory)

    model, mapping_dict_to_class = initialize_model(device,
                            num_layers = depth_transformer_encoder_blocks,
                            n_attention_heads = n_attention_heads,
                            d_model=d_model,
                            dropout_rate_1=dropout_rate_1,
                            dropout_rate_2=dropout_rate_2,
                            act_function = act_function,
                            transition_weight = transition_weight)

    #Define settings for training
    epochs = 20
    steps_per_epoch = len(train_data) / batch_size                               #The number of steps taken per epoch
    print("Steps per epoch: ", steps_per_epoch)
    eval_every_n_steps = 4000 #MODIFY to 4000
    print(f"Evaluating {round(steps_per_epoch/eval_every_n_steps, 1)} times per epoch")

    #Initialize the loss tracker
    tracker = CategoricalLossTracker(sequence_types)

    #Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_scratch)

    #Initialize variables for early stopping
    best_val_loss = float('inf')
    threshold_patience = 6  #Number of evaluations with no improvement to wait before stopping
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

        for i, batch in enumerate(train_loader):
            #Run training iteration
            epoch_losses = training_iteration(i, batch, scaler, model, optimizer, device, epoch_losses)

            #Validation step
            if step % eval_every_n_steps == 0 and step > 0:
                clear_memory()
                model.eval()

                #Initialize
                val_losses = []
                all_val_true_sequences = []
                all_val_pred_sequences = []
                all_val_sequence_types = [] 

                with torch.no_grad():
                    for counter, val_batch in enumerate(val_loader):
                        v_inputs_nt_rf0 = val_batch["nt_encodings_rf0"].to(device, non_blocking=True)
                        v_inputs_nt_rf1 = val_batch["nt_encodings_rf1"].to(device, non_blocking=True)
                        v_inputs_nt_rf2 = val_batch["nt_encodings_rf2"].to(device, non_blocking=True)

                        v_encoded_labels = val_batch['label_encodings'].to(device, non_blocking=True).long()

                        # Save padding mask before overwriting v_labels
                        padding_mask = (v_encoded_labels == -1)
                        valid_mask = ~padding_mask  # This is what we'll use consistently

                        # Forward pass for validation
                        if scaler is not None:
                            with autocast("cuda"):
                                v_outputs = model(v_inputs_nt_rf0, 
                                                v_inputs_nt_rf1,
                                                v_inputs_nt_rf2, 
                                                v_encoded_labels)
                                v_loss = v_outputs['loss']
                        else:
                            v_outputs = model(v_inputs_nt_rf0, 
                                            v_inputs_nt_rf1, 
                                            v_inputs_nt_rf2, 
                                            v_encoded_labels)
                            v_loss = v_outputs['loss']

                        val_losses.append(v_loss.item())

                        logits_for_metrics = v_outputs['logits']  #Store logits for decoding a few examples later

                        #Calculate category-specific losses (keep existing logic)
                        seq_descs_batch = val_batch['seq_desc']

                        for desc in tracker.categories:
                            #Find which sequences in the batch belong to this desc
                            desc_mask = torch.tensor([d == desc for d in seq_descs_batch], device=device)
                            if desc_mask.any():
                                #Select only the relevant sequences
                                desc_logits = v_outputs["logits"][desc_mask]           #[num_desc, seq_len, C]

                                #Use the same label preprocessing as the main loss
                                desc_labels_original = v_encoded_labels[desc_mask]     #Original labels
                                desc_valid_mask = valid_mask[desc_mask]                #Valid positions mask

                                #Calculate sequence type-specific loss
                                desc_crf_loss = -model.CRF.crf(desc_logits, desc_labels_original, 
                                                            mask=desc_valid_mask, reduction='mean')

                                tracker.update(desc, desc_crf_loss)

                                del desc_logits, desc_labels_original, desc_valid_mask, desc_crf_loss

                            del desc_mask

                        #Get all true labels and predicted labels from shared CRF space
                        for seq_idx in range(v_encoded_labels.shape[0]):
                            mask = ~padding_mask[seq_idx]  #Use NOT padding_mask to select valid positions
                            if mask.any():  #Skip empty sequences
                                true_seq = v_encoded_labels[seq_idx][mask].cpu().numpy()

                                #decode prediction for unmasked part of sequence
                                seq_logits = logits_for_metrics[seq_idx:seq_idx+1]  # [1, seq_len, num_classes]
                                seq_valid_mask = valid_mask[seq_idx:seq_idx+1]
                                pred_decoded = model.CRF.crf.decode(seq_logits, mask=seq_valid_mask)
                                pred_seq = np.array(pred_decoded[0])

                                seq_type = seq_descs_batch[seq_idx]

                                all_val_true_sequences.append(true_seq)
                                all_val_pred_sequences.append(pred_seq)
                                all_val_sequence_types.append(seq_type)

                        if counter == 0:
                            #Monitor hard examples as demonstration of how well classification is going
                            show_examples(v_encoded_labels, padding_mask, logits_for_metrics, seq_descs_batch, mapping_dict_to_class, model, device, valid_mask)

                        if counter % 30 == 0:
                            # Clean up each validation batch immediately
                            del v_inputs_nt_rf0, v_inputs_nt_rf1, v_inputs_nt_rf2
                            del v_encoded_labels, v_outputs, v_loss, padding_mask
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

                #Calculate performance metrics
                if all_val_true_sequences and all_val_pred_sequences:
                    sequence_metrics = calculate_sequence_accuracy_metrics(all_val_true_sequences, 
                                                                        all_val_pred_sequences,
                                                                        all_val_sequence_types)
                else:
                    sequence_metrics = {}

                #Calculate training loss based on batches from given training iteration
                if epoch_losses:
                    train_avg_loss = sum(epoch_losses[-eval_every_n_steps:]) / min(len(epoch_losses), eval_every_n_steps)
                else:
                    train_avg_loss = 0.0

                #Early stopping check
                if val_avg_loss < best_val_loss:
                    best_val_loss = val_avg_loss
                    torch.save(model.state_dict(), f"{input_data_dir_path}/models_optuna_codon_encoding/trial{trial.number}_trained.pth")
                    counter_patience = 0
                else:
                    counter_patience += 1

                if counter_patience >= threshold_patience:
                    print("Early stopping triggered!")
                    break

                #Log performance metrics
                val_times_counter = log_evaluation_metrics(epoch, train_avg_loss, val_avg_loss, best_val_loss, tracker, sequence_metrics, val_times_counter, sequence_types)

                #Clean up validation data
                del val_losses, all_val_true_sequences, all_val_pred_sequences
                clear_memory()

                model.train()  #Back to training mode


            # Increment global step counter after each batch
            step += 1

        #Check for early stopping again at end of epoch
        if counter_patience >= threshold_patience:
            break


    #############################################
    ###Final validation on full validation set###
    #############################################
    #Initialize model
    model = CDSPredictor(num_layers = depth_transformer_encoder_blocks,
                         n_attention_heads = n_attention_heads,
                         d_model = d_model,
                         dropout_rate_1 = dropout_rate_1, 
                         dropout_rate_2 = dropout_rate_2,
                         act_function = act_function,
                         transition_weight = transition_weight,
                         num_encoded_labels = len(mapping_dict_to_class.keys()),
                         encoded_labels_mapping = mapping_dict_to_class)

    model.to(device)
    #Load in parameters from trained model of best checkpoint
    model.load_state_dict(torch.load(f"{input_data_dir_path}/models_optuna_codon_encoding/trial{trial.number}_trained.pth", map_location=device), strict=False)

    model.eval()
    val_losses = []

    with torch.no_grad():
        for counter, val_batch in enumerate(val_loader_full):
            v_inputs_nt_rf0 = val_batch["nt_encodings_rf0"].to(device, non_blocking=True)
            v_inputs_nt_rf1 = val_batch["nt_encodings_rf1"].to(device, non_blocking=True)
            v_inputs_nt_rf2 = val_batch["nt_encodings_rf2"].to(device, non_blocking=True)
            v_encoded_labels = val_batch['label_encodings'].to(device, non_blocking=True).long()

            #Save padding mask before overwriting v_labels
            padding_mask = (v_encoded_labels == -1)
            valid_mask = ~padding_mask  #This is what we'll use consistently

            #Forward pass for validation
            if scaler is not None:
                with autocast("cuda"):
                    v_outputs = model(v_inputs_nt_rf0,
                                    v_inputs_nt_rf1,
                                    v_inputs_nt_rf2, 
                                    v_encoded_labels)
                    v_loss = v_outputs['loss']
            else:
                v_outputs = model(v_inputs_nt_rf0, 
                                v_inputs_nt_rf1,
                                v_inputs_nt_rf2,
                                v_encoded_labels)
                v_loss = v_outputs['loss']

            val_losses.append(v_loss.item())

            #Process category-specific losses (keep existing logic)
            seq_descs_batch = val_batch['seq_desc']

            #Calculate category-specific losses (keep existing logic)
            seq_descs_batch = val_batch['seq_desc']

            for desc in tracker.categories:
                #Find which sequences in the batch belong to this desc
                desc_mask = torch.tensor([d == desc for d in seq_descs_batch], device=device)
                if desc_mask.any():
                    #Select only the relevant sequences
                    desc_logits = v_outputs["logits"][desc_mask]           #[num_desc, seq_len, C]

                    #Use the same label preprocessing as the main loss
                    desc_labels_original = v_encoded_labels[desc_mask]     #Original labels
                    desc_valid_mask = valid_mask[desc_mask]                #Valid positions mask

                    #Calculate sequence type-specific loss
                    desc_crf_loss = -model.CRF.crf(desc_logits, desc_labels_original, 
                                                mask=desc_valid_mask, reduction='mean')

                    tracker.update(desc, desc_crf_loss)

                    del desc_logits, desc_labels_original, desc_valid_mask, desc_crf_loss

                del desc_mask

            if counter % 30 == 0:
                #Clean up each validation batch immediately
                del v_inputs_nt_rf0, v_inputs_nt_rf1,v_inputs_nt_rf2, v_encoded_labels, v_outputs, v_loss, padding_mask
                clear_memory()

    #Calculate validation loss and other final metrics for validation loop
    if val_losses:
        val_avg_loss = sum(val_losses) / len(val_losses)
    else:
        val_avg_loss = float('inf')

    wandb_log = {}

    wandb_log[f"full_val_loss"] = val_avg_loss

    #Add loss metrics for each sequence type
    for seq_type in sequence_types:
        wandb_log[f"full_val_loss_{seq_type}"] = tracker.get_metrics().get(seq_type, 0)

    #Log to wandb
    wandb.log(wandb_log)
    wandb.finish()

    return best_val_loss


######################################################################################################################################################################################################
############################################################################################## Main ##################################################################################################
######################################################################################################################################################################################################

np.random.seed(42)

#Load in data once
train_data, val_data, sequence_types, seq_type_desc_fracs = load_and_process_data(max_len)
val_loader_full = load_full_validation_set(max_len)

#Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize',   #Minimize loss
                            pruner=optuna.pruners.HyperbandPruner())
study.optimize(
    lambda trial: objective(trial, train_data, val_data, val_loader_full, sequence_types, seq_type_desc_fracs), 
    n_trials=30
)

model_config = {
    "depth_transformer_encoder_blocks": study.best_params['depth_transformer_encoder_blocks'],
    "n_attention_heads": study.best_params['n_attention_heads'],
    "d_model": study.best_params['d_model'],
    "dropout_rate_1": study.best_params['dropout_rate_1'],
    "dropout_rate_2": study.best_params['dropout_rate_2'],
    "lr_scratch": study.best_params['lr_scratch'],
    "act_function": study.best_params['act_function'],
    "transition_weight": study.best_params["transition_weight"]}

# Wrap in hyperparameters key for Hydra structure
config = {"hyperparameters": model_config}

#Save the dictionary as a YAML file
file_path = f'{input_data_dir_path}/hyperparameter_configs/codon_encoding_hyperparameters.yaml'
with open(file_path, 'w') as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)

print(f"Model configuration saved to {file_path}.")

