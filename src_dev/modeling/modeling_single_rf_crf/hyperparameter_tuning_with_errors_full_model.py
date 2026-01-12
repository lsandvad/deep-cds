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

device = torch.device("cuda:1" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device_type = device.type  # "cuda", "mps", or "cpu"

if device_type == "mps":
    input_data_dir_path = "../../../data/processed_data/model_data/single_rf_crf/model_with_errors"
    num_workers_cpu = 0 #Adjust if not using mps
    pin_memory = False #Turn true if using CUDA
elif device_type == "cuda":
    input_data_dir_path = "/tmp/nrt204/FragmentPredictor3/data/processed_data/model_data/single_rf_crf/model_with_errors" #TEST ON SCARB CLUSTER
    num_workers_cpu = 4
    pin_memory = True
print("Device type:", device_type, flush = True)

#Model choice
esm2_model = "facebook/esm2_t6_8M_UR50D"
esm2_model_abbr = esm2_model.split("/")[-1].split("_UR")[0]
project_name = esm2_model_abbr

#dir in wandb to place run
wandb_project_name = "experiments_single_more_data"# "tune_single_full_model_updated" #MODIFY!

#Make sure dir to store model exists
os.makedirs(f"../../../data/processed_data/model_data/single_rf_crf/model_with_errors/models_optuna/", exist_ok=True)

max_aa_len = 100
max_len = max_aa_len + 2 #Add CLS and EOS tokens

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
    Dataset class for single reading frame sequence data.
    
    Stores nucleotide encodings, amino acid encodings, and labels for one reading 
    frame (rf0, rf1, rf2) along with sequence descriptions.
    
    Args:
        nt_encodings (list): List of nucleotide codon encodings (max_aa_len, 12) for reading frame
        aa_encodings (BatchEncoding): Tokenized amino acid encodings (dict) for reading frame
        labels (list): List of numpy arrays (int8) with per-position labels for reading frame
        seq_desc (list): List of sequence description strings/identifiers
        
    Returns:
        Dictionary containing all encodings and labels for a single sequence across
        all three reading frames, with tensors converted to appropriate dtypes.
    """
    def __init__(self, nt_encodings, aa_encodings, labels, seq_desc):
        self.nt_encodings = nt_encodings
        self.aa_encodings = aa_encodings
        self.labels = labels  
        self.seq_desc = seq_desc

    def __getitem__(self, idx):
        # Handle labels properly
        if isinstance(self.labels[idx], np.ndarray):
            labels_tensor = torch.from_numpy(self.labels[idx])
        elif isinstance(self.labels[idx], torch.Tensor):
            labels_tensor = self.labels[idx] 
        else:
            labels_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)

        item = {
            'aa_encodings': {
                key: val[idx] for key, val in self.aa_encodings.items()
            },
            'nt_encodings': torch.as_tensor(self.nt_encodings[idx], dtype=torch.float32),
            'labels': labels_tensor,
            'seq_desc': self.seq_desc[idx]
        }
        return item

    def __len__(self):
        return len(self.labels)



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
    nt_encodings = process_nt_sequences_to_codons(nt_sequences, max_aa_len)

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



def balance_training_samples(dataset_df):
    """ 
    Down-sample sequences so that there is equally many sequences that are: 
        1. Fully non-coding
        3. Coding or contains a transition (due to start/end of CDS or an indel error causing a frameshift)

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
                - "non_coding" if all labels are 0 (fully non-coding)
                - "some_coding" if the sequence contains a mix of labels (i.e., contains a transition)
        """
        unique_labels = set(label_seq)  #Get unique label values in the sequence
        unique_labels = {int(x) for x in unique_labels if str(x).isdigit()}  #Ensure labels are integers
        if unique_labels == {0}:
            return "non_coding"  #Fully non-coding
        else:
            return "some_coding"  #Contains a signal change (transition caused by indels or start/stop codons, substitution errors) (mixed labels)

    #Apply classification
    dataset_df["seq_type"] = dataset_df["aa_labels"].apply(classify_sequence)

    #Count how many transition sequences there are
    coding_sampled = len(dataset_df[dataset_df["seq_type"] == "some_coding"])

    #Sample that many from each of the flat types (or fewer if you want more imbalance)
    nc_sampled = dataset_df[dataset_df["seq_type"] == "non_coding"].sample(n=min(coding_sampled*4, len(dataset_df[dataset_df["seq_type"] == "non_coding"])), random_state=42)

    #Keep all transition sequences
    coding_sampled = dataset_df[dataset_df["seq_type"] == "some_coding"]

    #Combine them
    balanced_df = pd.concat([coding_sampled, nc_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    print("Training data samples, balanced: ", balanced_df.shape[0])

    return balanced_df


def load_and_process_data(max_len):
    """
    Main function that loads and processes all data efficiently.
    """
    # Load data
    train_set = pd.read_csv(
        f"{input_data_dir_path}/datasets_model/train_100_genomes.csv.gz",
        index_col=None, 
        compression="gzip"
    )
    val_set = pd.read_csv(
        f"{input_data_dir_path}/datasets_model/val.csv.gz",
        index_col=None, 
        compression="gzip"
    )

    seq_type_desc_fracs = (train_set['aa_desc'].value_counts(normalize=True)).to_dict()

    #Create a combined stratification label
    val_set["accession_seq_desc_merged"] = val_set["accession"].astype(str) + "_" + val_set["aa_desc"].astype(str)
    train_set["accession_seq_desc_merged"] = train_set["accession"].astype(str) + "_" + train_set["aa_desc"].astype(str) #DELETE

    #Sample 1% from each stratum
    val_set = val_set.groupby("accession_seq_desc_merged", group_keys=False).sample(frac=0.005, random_state=42) #MODIFY #approx. 140k validation samples -> frac = 0.02
    train_set = train_set.groupby("accession_seq_desc_merged", group_keys=False).sample(frac=0.05, random_state=42) #DELETE; remove

    print("Training data samples: ", train_set.shape[0])
    print("Validation data samples during training: ", val_set.shape[0])
    print("Distribution of sequence types in training set, before balancing: ", seq_type_desc_fracs)

    # Create tokenizer once and reuse
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esm2_t6_8M_UR50D",
        do_lower_case=False,
    )

    # Process training data
    #balanced_train = balance_training_samples(train_set)
    train_data, sequence_types = encode_data(train_set, max_len, tokenizer)

    seq_type_desc_fracs_downsampled = (train_set['aa_desc'].value_counts(normalize=True)).to_dict()

    # Clear intermediate data to save memory
    #del balanced_train

    # Process validation data
    val_data, _ = encode_data(val_set, max_len, tokenizer)

    return train_data, val_data, sequence_types, seq_type_desc_fracs_downsampled


def load_full_validation_set(max_len):
    """ 
    Load full validation set for final validation
    """
    # Load data
    val_set = pd.read_csv(
        f"{input_data_dir_path}/datasets_model/val.csv.gz",
        index_col=None, 
        compression="gzip"
    )

    #Create a combined stratification label
    val_set["accession_seq_desc_merged"] = val_set["accession"].astype(str) + "_" + val_set["aa_desc"].astype(str) #DELETE

    #Sample 1% from each stratum
    val_set = val_set.groupby("accession_seq_desc_merged", group_keys=False).sample(frac=0.02, random_state=42) #DELETE

    print("Validation data samples in full set: ", val_set.shape[0])

    # Create tokenizer once and reuse
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esm2_t6_8M_UR50D",
        do_lower_case=False,
    )

    # Process validation data
    val_data, _ = encode_data(val_set, max_len, tokenizer)

    val_loader = DataLoader(
        val_data, 
        batch_size=100, 
        shuffle=False, 
        num_workers=num_workers_cpu,  
        pin_memory=pin_memory)

    return val_loader


#Model Framework


#Define model architecture 
class SequenceEncoder(nn.Module):
    """
    Sequence encoder using the pretrained ESM-2.
    
    Args:
        esm2_model (str): Name or path of the pretrained ESM-2 model to load
        dropout_rate_1 (float): Dropout rate for regularization layer applied after ESM-2
    
    Attributes:
        pretrained_model_aa (AutoModel): Pretrained ESM-2 model with all layers initially frozen
        dropout_1 (nn.Dropout): Dropout layer for regularization before transformer head
    """
    def __init__(self,
                 esm2_model,
                 dropout_rate_1): 
        super(SequenceEncoder, self).__init__()

        #Load pretrained ESM-2 model for amino acid sequences
        self.pretrained_model_aa = AutoModel.from_pretrained(esm2_model)
        
        self.num_layers = len(self.pretrained_model_aa.encoder.layer)

        #Freeze ALL layers of ESM-2 initially for staged training
        for param in self.pretrained_model_aa.parameters():
            param.requires_grad = False

        #Additional dropout layer for regularization after encoding sequences
        self.dropout_1 = nn.Dropout(dropout_rate_1)
        
        # Track which layers to unfreeze later
        self.unfreeze_start = None

    def prepare_for_unfreezing(self, unfreeze_fraction=0.5):
        """
        Prepare to unfreeze top layers later. This just calculates which layers
        will be unfrozen but doesn't actually unfreeze them yet.
        
        Args:
            unfreeze_fraction: Fraction of top layers to eventually unfreeze (default 0.5 = top half)
        """
        self.unfreeze_start = int(self.num_layers * (1 - unfreeze_fraction))
        print(f"Prepared to unfreeze layers {self.unfreeze_start} to {self.num_layers-1} (top {unfreeze_fraction*100}%)")

    def unfreeze_top_layers(self, lr_esm2, optimizer, warmup_factor=0.1):
        """
        Unfreeze the top layers and add them to the optimizer with warmup.
        
        Args:
            lr_esm2: Learning rate for ESM-2 layers
            optimizer: The optimizer to add parameters to
            warmup_factor: Initial LR will be lr_esm2 * warmup_factor (default 0.1 = 10% of target)
            
        Returns:
            int: Index of the new parameter group in the optimizer
        """
        if self.unfreeze_start is None:
            raise ValueError("Must call prepare_for_unfreezing() before unfreeze_top_layers()")
        
        # Unfreeze the top layers
        unfrozen_count = 0
        for i, layer in enumerate(self.pretrained_model_aa.encoder.layer):
            if i >= self.unfreeze_start:
                for param in layer.parameters():
                    param.requires_grad = True
                    unfrozen_count += 1
        
        print(f"Unfrozen {unfrozen_count} parameters in layers {self.unfreeze_start} to {self.num_layers-1}")
        
        # Collect parameters from unfrozen layers
        unfrozen_params = [
            p for layer in self.pretrained_model_aa.encoder.layer[self.unfreeze_start:] 
            for p in layer.parameters()
        ]
        
        # Add to optimizer with warmup (start at reduced LR)
        initial_lr = lr_esm2 * warmup_factor
        optimizer.add_param_group({
            'params': unfrozen_params, 
            'lr': initial_lr
        })
        
        param_group_idx = len(optimizer.param_groups) - 1
        print(f"Added {len(unfrozen_params)} parameters to optimizer with initial LR={initial_lr:.2e} (warmup factor={warmup_factor})")
        
        return param_group_idx
    
    def warmup_lr(self, optimizer, param_group_idx, target_lr, current_step, warmup_steps):
        """
        Gradually increase learning rate for unfrozen layers during warmup period.
        
        Args:
            optimizer: The optimizer
            param_group_idx: Index of the parameter group to adjust
            target_lr: Target learning rate to reach
            current_step: Current step in warmup (0 to warmup_steps)
            warmup_steps: Total warmup steps
        """
        if current_step >= warmup_steps:
            new_lr = target_lr
        else:
            # Linear warmup
            warmup_factor = current_step / warmup_steps
            new_lr = target_lr * warmup_factor
        
        optimizer.param_groups[param_group_idx]['lr'] = new_lr

    def forward(self, x_aa, attention_mask_aa):
        """
        Forward pass through the sequence encoder.
    
        Args:
            x_aa (torch.Tensor): Tokenized amino acid sequences of shape (batch_size, seq_len)
            attention_mask_aa (torch.Tensor): Attention mask of shape (batch_size, seq_len)
            
        Returns:
            embeddings_aa (torch.Tensor): Amino acid embeddings with CLS/EOS removed, shape (batch_size, seq_len-2, hidden_size)
            attention_mask_trimmed (torch.Tensor): Attention mask with CLS/EOS removed, shape (batch_size, seq_len-2)
        """
        #Extract features from pretrained ESM-2 model
        features_aa = self.pretrained_model_aa(x_aa, attention_mask=attention_mask_aa)

        #Get last hidden state: [batch_size, tokens, hidden_size]
        sequence_output_aa = features_aa['last_hidden_state']

        #Remove CLS and EOS token embeddings: [batch_size, aa_seq_len, hidden_size]
        sequence_output_aa = sequence_output_aa[:, 1:-1, :]

        #Apply dropout before transformer head
        embeddings_aa = self.dropout_1(sequence_output_aa)

        #Remove CLS/EOS from attention mask for transformer head
        attention_mask_trimmed = attention_mask_aa[:, 1:-1]

        return embeddings_aa, attention_mask_trimmed

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
        combined_codon_and_aa_embeddings = torch.cat([encoded_embeddings_aa, encoded_seqs_nt], dim=-1)

        combined_codon_and_aa_embeddings = combined_codon_and_aa_embeddings.permute(1, 0, 2)  # [seq_len, batch, hidden + 3*4]
        attention_mask_transformer = ~trimmed_attention_mask.bool()

        combined_codon_and_aa_embeddings = self.encoder(combined_codon_and_aa_embeddings, src_key_padding_mask=attention_mask_transformer)

        combined_codon_and_aa_embeddings = combined_codon_and_aa_embeddings.permute(1, 0, 2)  # [batch, seq_len, hidden]

        combined_codon_and_aa_embeddings = self.norm(combined_codon_and_aa_embeddings)

        logits = self.linear(combined_codon_and_aa_embeddings)  # [batch, seq_len, num_labels]

        return logits


class LinearChainCRF(nn.Module):
    """
    Neural network module that adds a CRF layer for structured prediction.
    """
    def __init__(self, 
                 num_labels, 
                 transition_weight):
        super().__init__()
        self.crf = CRF(num_tags=num_labels, batch_first=True)
        
        # Define legal transitions as adjacency list
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
        
        # Three-tier initialization
        with torch.no_grad():
            # Stage 1: All illegal transitions → -10 (forbidden)
            self.crf.transitions[~self.biologically_valid_mask] = -10
            
            # Stage 2: Legal but infrequent transitions → transition_weight + small noise (discouraged but diverse)
            legal_infrequent = self.biologically_valid_mask & ~self.frequent_transition_mask
            noise = torch.randn_like(self.crf.transitions[legal_infrequent]) * 0.1
            self.crf.transitions[legal_infrequent] = transition_weight + noise
            
            # Stage 3: Frequent self-transitions (0→0, 1→1) → 0 (neutral/encouraged)
            self.crf.transitions[self.frequent_transition_mask] = 0
    
    def _create_biologically_valid_mask(self):
        """Create mask from legal_transitions dictionary."""
        # Start with all transitions forbidden
        self.biologically_valid_mask.fill_(False)
        
        # Mark legal transitions as True
        for from_label, to_labels in self.legal_transitions.items():
            for to_label in to_labels:
                self.biologically_valid_mask[from_label, to_label] = True
    
    def _create_frequent_transition_mask(self):
        """
        Create a mask for frequent self-transitions (0→0 and 1→1).
        """
        # Mark 0→0 and 1→1 as frequent
        self.frequent_transition_mask[0, 0] = True
        self.frequent_transition_mask[1, 1] = True
        
        print(f"Frequent self-transitions: 0→0, 1→1")
    
    def forward(self, logits, attention_mask, labels=None):
        if labels is not None:
            crf_mask = (labels != 9)  # True for valid positions, False for padding
            log_likelihood = self.crf(logits, labels, mask=crf_mask, reduction='none')
            
            loss = -log_likelihood.mean()
            
            return {'loss': loss, 'logits': logits}
        else:
            # Inference mode: decode best sequence
            crf_mask = (labels != 9)  # True for valid positions, False for padding
            predictions = self.crf.decode(logits, mask=crf_mask)
            return {'predictions': predictions, 'logits': logits}


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
                 act_function,
                 transition_weight): 
        super(CDSPredictor, self).__init__()

        self.sequence_encoder = SequenceEncoder(
            esm2_model,
            dropout_rate_1)

        self.TransformerEncoderBlock = TransformerEncoderBlock(
            hidden_size=self.sequence_encoder.pretrained_model_aa.config.hidden_size,
            num_layers=num_layers,
            n_attention_heads=n_attention_heads,
            dropout_rate_encoder=dropout_rate_2,
            act_function=act_function,
            num_labels=6)

        self.CRF = LinearChainCRF(num_labels=6,
                                  transition_weight = transition_weight)

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


def initialize_model(device, num_layers, n_attention_heads, dropout_rate_1, dropout_rate_2, act_function, transition_weight):
    """
    Initialize the model and move it to the specified device.
    Args:
        device_type (str): The device to use for computation ("cuda", "mps", or "cpu").
    Returns:
        device (torch.device): The device being used.
        model (nn.Module): The initialized model.
    """

    print("Running on: ", device, flush = True)

    model = CDSPredictor(esm2_model=esm2_model,
                         num_layers = num_layers,
                         n_attention_heads = n_attention_heads,
                         dropout_rate_1 = dropout_rate_1, 
                         dropout_rate_2 = dropout_rate_2,
                         act_function = act_function,
                         transition_weight = transition_weight)
    model.to(device)

    if device == "cuda":
        print(f"Memory Allocated after loading model: {torch.cuda.memory_allocated(device) / 1024**3} GB")

    return model


def print_model_dimensions(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



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


def log_evaluation_metrics(epoch, train_avg_loss, val_avg_loss, best_val_loss, tracker, sequence_metrics, val_times_counter, sequence_types):
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

    # Add loss metrics for each sequence type
    for seq_type in sequence_types:
        wandb_log[f"val_loss_{seq_type}"] = tracker.get_metrics().get(seq_type, 0)

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

    wandb_log[f"best_val_loss"] = best_val_loss

    # Log to wandb
    wandb.log(wandb_log)

    return val_times_counter + 1


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



def training_iteration(i, batch, scaler, model, optimizer, device, epoch_losses, warmup_state=None):
    # Move data to device
    inputs_nt = batch["nt_encodings"].to(device, non_blocking=True)
    inputs_aa = batch["aa_encodings"]["input_ids"].to(device, non_blocking=True)
    attention_mask_aa = batch["aa_encodings"]["attention_mask"].to(device, non_blocking=True)
    labels = batch['labels'].to(device, non_blocking=True).long()

    # Forward pass - use mixed precision if available
    if scaler is not None:
        with autocast("cuda"):
            outputs = model(inputs_nt, inputs_aa, attention_mask_aa, 
                          labels=labels)
            loss = outputs['loss']
    else:
        # Regular forward pass without mixed precision
        outputs = model(inputs_nt, inputs_aa, attention_mask_aa, 
                       labels=labels)
        loss = outputs['loss']

    if scaler is not None:
        #Mixed precision backward pass
        scaler.scale(loss).backward()
        
        #Unscale before gradient clipping
        scaler.unscale_(optimizer)
        
        #Gradient clipping (CRITICAL for CRF stability!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
    else:
        #Regular backward pass
        loss.backward()
        
        #Gradient clipping (CRITICAL for CRF stability!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

    #Handle warmup if ESM-2 layers were unfrozen
    if warmup_state is not None and warmup_state['counter'] < warmup_state['total_steps']:
        model.sequence_encoder.warmup_lr(
            optimizer, 
            warmup_state['param_group_idx'], 
            warmup_state['target_lr'], 
            warmup_state['counter'], 
            warmup_state['total_steps']
        )
        warmup_state['counter'] += 1

    # Store loss value and cleanup immediately
    epoch_losses.append(loss.item())

    # Print weights in transition matrix
    if i % 10000 == 0:
        print(model.CRF.crf.transitions)

    if i % 1000 == 0:
        # Cleanup to prevent memory leaks
        del outputs, loss
        del inputs_nt, inputs_aa, attention_mask_aa, labels

    return epoch_losses, warmup_state


def show_examples(v_labels, padding_mask, predictions, seq_descs_batch):
    # Show hard examples as demonstration
    for seq_i in range(min(32, v_labels.shape[0])):
        mask = ~padding_mask[seq_i]
        if seq_descs_batch[seq_i] not in ["non-coding", "coding", "coding_with_substitutions"]:
            print("Sequence type:", seq_descs_batch[seq_i])
            print("Labels:\n", v_labels[seq_i][mask].cpu().numpy().astype(float))
            print("Predictions:\n", predictions[seq_i][mask].cpu().numpy().astype(float))
            print("\n")

    for seq_i in range(min(3, v_labels.shape[0])):
        mask = ~padding_mask[seq_i]
        if seq_descs_batch[seq_i] in ["non-coding", "coding", "coding_with_substitutions"]:
            print("Sequence type:", seq_descs_batch[seq_i])
            print("Labels:\n", v_labels[seq_i][mask].cpu().numpy().astype(float))
            print("Predictions:\n", predictions[seq_i][mask].cpu().numpy().astype(float))
            print("\n")


def objective(trial, train_data, val_data, val_loader_full, sequence_types, seq_type_desc_fracs):
    #Define hyperparameter ranges to sample from
    depths_transformer_encoder_blocks = [2, 4, 6, 8]
    attention_heads = [2, 4]
    dropout_rates_1 = [0.1, 0.2, 0.3] #Dropout rate applied after ESM-2 encoding
    dropout_rates_2 = [0.2, 0.3, 0.4, 0.5] #dropout rate applied in transformer encoder layers
    act_functions = ["relu", "gelu"]

    #Define trial suggestions; set hyperparameters
    depth_transformer_encoder_blocks = trial.suggest_categorical('depth_transformer_encoder_blocks', depths_transformer_encoder_blocks)
    n_attention_heads = trial.suggest_categorical('n_attention_heads', attention_heads)
    dropout_rate_1 = trial.suggest_categorical('dropout_rate_1', dropout_rates_1)
    dropout_rate_2 = trial.suggest_categorical('dropout_rate_2', dropout_rates_2)
    lr_scratch = trial.suggest_float('lr_scratch', 5e-6, 8e-5, log=True)
    lr_esm2 = lr_scratch * trial.suggest_float('lr_fraction', 0.01, 0.1, log=True)
    act_function = trial.suggest_categorical('act_function', act_functions)
    transition_weight = trial.suggest_float('transition_weight', -4, -1)

    batch_size = 32


    print(f"depth_transformer_encoder_blocks: {depth_transformer_encoder_blocks}\n \
            n_attention_heads: {n_attention_heads}\n \
            dropout_rate_1: {dropout_rate_1}\n \
            dropout_rate_2: {dropout_rate_2}\n \
            lr_esm2: {lr_esm2}\n \
            lr_scratch: {lr_scratch}\n \
            transition_weight: {transition_weight}\n \
            act_function: {act_function}")

    wandb.init(project=wandb_project_name, 
                config={
                        "depth_transformer_encoder_blocks": depth_transformer_encoder_blocks,
                        "n_attention_heads": n_attention_heads,
                        "dropout_rate_1": dropout_rate_1,
                        "dropout_rate_2": dropout_rate_2,
                        "act_function": act_function,
                        "transition_weight": transition_weight,
                        "lr_esm2": lr_esm2,
                        "lr_scratch": lr_scratch},
                        name = f"trial_{trial.number}")

    type_weights = {st: 1.0 for st in sequence_types}  # Initial sampling weights

    # Create initial weighted sampler
    train_sampler = create_weighted_sampler(train_data, type_weights)

    #Define data loaders
    train_loader = adjust_train_sample_distribution(train_data, train_sampler, batch_size)

    val_loader = DataLoader(
        val_data, 
        batch_size=100, #larger batch size for validation 
        shuffle=True, 
        num_workers=num_workers_cpu,  
        pin_memory=pin_memory)

    model = initialize_model(device,
                            num_layers = depth_transformer_encoder_blocks,
                            n_attention_heads = n_attention_heads,
                            dropout_rate_1=dropout_rate_1,
                            dropout_rate_2=dropout_rate_2,
                            act_function = act_function,
                            transition_weight = transition_weight)
    # Prepare for unfreezing later (but don't unfreeze yet!)
    model.sequence_encoder.prepare_for_unfreezing(unfreeze_fraction=0.5)


    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    #print(f"Model dimensions: {print_model_dimensions(model)}")

    #Define settings for training
    epochs = 10
    steps_per_epoch = len(train_data) / batch_size                               #The number of steps taken per epoch
    print("Steps per epoch: ", steps_per_epoch)
    eval_every_n_steps = 1500 #MODIFY TO 4000!!!
    print(f"Evaluating {round(steps_per_epoch/eval_every_n_steps, 1)} times per epoch")
    freeze_esm_validations = 15  #unfreeze after approximately 1 epoch = 15 validations (2.4M training samples)


    #Initialize the loss tracker
    tracker = CategoricalLossTracker(sequence_types)

    # Initialize optimizer - NO ESM-2 parameters at all initially
    optimizer = torch.optim.AdamW([
        {'params': model.TransformerEncoderBlock.parameters(), 'lr': lr_scratch},
        {'params': model.CRF.parameters(), 'lr': lr_scratch}
    ])
            

    #Initialize variables for early stopping
    best_val_loss = float('inf')
    threshold_patience = 6  #Number of evaluations with no improvement to wait before stopping
    counter_patience = 0

    #Initialize variables for training loop
    step = 0  # Global step counter
    val_times_counter = 0


    # Initialize warmup state (will be set when unfreezing)
    warmup_state = None

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
            epoch_losses, warmup_state = training_iteration(i, batch, scaler, model, optimizer, device, epoch_losses, warmup_state)

            #Validation step
            if step % eval_every_n_steps == 0 and step > 0:
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
                        #v_labels_modified = v_labels_original.clone()
                        #v_labels_modified[padding_mask] = 0  # Set padding to valid label for forward pass


                        # Forward pass for validation
                        if scaler is not None:
                            with autocast("cuda"):
                                v_outputs = model(v_inputs_nt, v_inputs_aa, v_attention_aa, v_labels_original)
                                v_loss = v_outputs['loss']
                        else:
                            v_outputs = model(v_inputs_nt, v_inputs_aa, v_attention_aa, v_labels_original)
                            v_loss = v_outputs['loss']

                        val_losses.append(v_loss.item())

                        # Get predictions and process sequences immediately
                        predictions = model.CRF.crf.decode(v_outputs['logits'], mask=valid_mask)

                        # Pad predictions to match v_labels shape for downstream code
                        # efficient - create tensor directly and pad
                        batch_size, seq_len = v_labels_original.shape
                        predictions_tensor = torch.full((batch_size, seq_len), 9, 
                                                        dtype=torch.long, device=v_labels_original.device)

                        for i_pred, seq in enumerate(predictions):
                            seq_len_actual = len(seq)
                            if seq_len_actual > 0:
                                predictions_tensor[i_pred, :seq_len_actual] = torch.tensor(seq, 
                                                                                    dtype=torch.long, 
                                                                                    device=v_labels_original.device)
                        predictions = predictions_tensor

                        # Process category-specific losses (keep your existing logic)
                        seq_descs_batch = val_batch['seq_desc']
                        seq_len = v_inputs_aa.size(1) - 2

                        for desc in tracker.categories:
                            desc_mask = torch.tensor([d == desc for d in seq_descs_batch], device=device)
                            if desc_mask.any():
                                desc_logits = v_outputs["logits"][desc_mask]
                                desc_labels = v_labels_original[desc_mask]  # Use desc_labels, don't reassign v_labels_original
                                desc_valid_mask = valid_mask[desc_mask]
                                
                                desc_crf_loss = -model.CRF.crf(desc_logits, desc_labels, 
                                                            mask=desc_valid_mask, reduction='mean')
                                
                                tracker.update(desc, desc_crf_loss)
                                
                                del desc_logits, desc_labels, desc_valid_mask, desc_crf_loss
                            
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

                        if counter == 0 or counter == 1:
                            #Monitor hard examples as demonstration of how well classification is going
                            show_examples(v_labels_original, padding_mask, predictions, seq_descs_batch)

                        if counter % 30 == 0: #clean up every 12600 samples (30 batches of size 420)
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
                    train_avg_loss = sum(epoch_losses[-eval_every_n_steps:]) / min(len(epoch_losses), eval_every_n_steps)
                else:
                    train_avg_loss = 0.0

                # Early stopping check
                if val_avg_loss < best_val_loss:
                    best_val_loss = val_avg_loss
                    torch.save(model.state_dict(), f"../../../data/processed_data/model_data/single_rf_crf/model_with_errors/models_optuna/trial{trial.number}_trained_test.pth")
                    counter_patience = 0
                else:
                    counter_patience += 1

                if counter_patience >= threshold_patience:
                    print("Early stopping triggered!")
                    break

                #Log metrics
                val_times_counter = log_evaluation_metrics(epoch, train_avg_loss, val_avg_loss, best_val_loss, tracker, sequence_metrics, val_times_counter, sequence_types)


                if val_times_counter == freeze_esm_validations:
                    # Unfreeze ESM-2 layers and set up warmup
                    esm2_param_group_idx = model.sequence_encoder.unfreeze_top_layers(
                        lr_esm2=lr_esm2, 
                        optimizer=optimizer, 
                        warmup_factor=0.1  # Start at 10% of lr_esm2
                    )
                    # Initialize warmup state that will be used in training_iteration
                    warmup_state = {
                        'counter': 0,
                        'total_steps': 1000,  # Warmup over 1000 steps
                        'param_group_idx': esm2_param_group_idx,
                        'target_lr': lr_esm2
                    }

                #Clean up validation data
                del val_losses, all_val_true_sequences, all_val_pred_sequences
                clear_memory()

                model.train()  # Back to training mode

            # Increment global step counter after each batch
            step += 1

        # Check if early stopping was triggered during validation
        if counter_patience >= threshold_patience:
            break

    #############################################
    ###Final validation on full validation set###
    #############################################
    model = CDSPredictor(esm2_model=esm2_model,
                        num_layers=depth_transformer_encoder_blocks,
                        n_attention_heads=n_attention_heads,
                        dropout_rate_1=dropout_rate_1,
                        dropout_rate_2=dropout_rate_2,
                        act_function=act_function,
                        transition_weight = transition_weight)


    model.to(device)
    #Load in parameters from trained model
    model.load_state_dict(torch.load(f"../../../data/processed_data/model_data/single_rf_crf/model_with_errors/models_optuna/trial{trial.number}_trained_test.pth", map_location=device), strict=False)
    #After loading the model
    #model = torch.compile(model)

    model.eval()
    val_losses = []

    with torch.no_grad():
        for counter, val_batch in enumerate(val_loader_full):
            v_inputs_nt = val_batch["nt_encodings"].to(device, non_blocking=True)
            v_inputs_aa = val_batch["aa_encodings"]["input_ids"].to(device, non_blocking=True)
            v_attention_aa = val_batch["aa_encodings"]["attention_mask"].to(device, non_blocking=True)
            v_labels_original = val_batch['labels'].to(device, non_blocking=True).long()

            # Save padding mask before overwriting v_labels
            padding_mask = (v_labels_original == 9)
            valid_mask = ~padding_mask  # This is what we'll use consistently

            # Set all padding positions to a valid label (e.g., 0)
            # Create modified labels for model forward pass
            #v_labels_modified = v_labels_original.clone()
            #v_labels_modified[padding_mask] = 0  # Set padding to valid label for forward pass

            # Forward pass for validation
            if scaler is not None:
                with autocast("cuda"):
                    v_outputs = model(v_inputs_nt, v_inputs_aa, v_attention_aa, v_labels_original)
                    v_loss = v_outputs['loss']
            else:
                v_outputs = model(v_inputs_nt, v_inputs_aa, v_attention_aa, v_labels_original)
                v_loss = v_outputs['loss']

            val_losses.append(v_loss.item())

            # Process category-specific losses (keep your existing logic)
            seq_descs_batch = val_batch['seq_desc']
            seq_len = v_inputs_aa.size(1) - 2

            for desc in tracker.categories:
                desc_mask = torch.tensor([d == desc for d in seq_descs_batch], device=device)
                if desc_mask.any():
                    desc_logits = v_outputs["logits"][desc_mask]
                    desc_labels = v_labels_original[desc_mask]  # Use desc_labels, don't reassign v_labels_original
                    desc_valid_mask = valid_mask[desc_mask]
                    
                    desc_crf_loss = -model.CRF.crf(desc_logits, desc_labels, 
                                                mask=desc_valid_mask, reduction='mean')
                    
                    tracker.update(desc, desc_crf_loss)
                    
                    del desc_logits, desc_labels, desc_valid_mask, desc_crf_loss
                
                del desc_mask

            if counter % 50 == 0: #clean up every 12800 samples (50 batches of size 256)
                # Clean up each validation batch immediately
                del v_inputs_nt, v_inputs_aa, v_attention_aa, v_labels_original, v_outputs, v_loss
                del padding_mask
                clear_memory()

    # Calculate validation loss and other final metrics for validation loop
    if val_losses:
        val_avg_loss = sum(val_losses) / len(val_losses)
    else:
        val_avg_loss = float('inf')

    wandb_log = {}

    wandb_log[f"full_val_loss"] = val_avg_loss

    # Add loss metrics for each sequence type
    for seq_type in sequence_types:
        wandb_log[f"full_val_loss_{seq_type}"] = tracker.get_metrics().get(seq_type, 0)

    # Log to wandb
    wandb.log(wandb_log)
    ###

    wandb.finish()

    return best_val_loss


# # Main code
np.random.seed(42)

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
    "dropout_rate_1": study.best_params['dropout_rate_1'],
    "dropout_rate_2": study.best_params['dropout_rate_2'],
    "lr_esm2": study.best_params['lr_esm2'],
    "lr_scratch": study.best_params['lr_scratch'],
    "act_function": study.best_params['act_function'],
    "transition_weight": study.best_params["transition_weight"]}

os.makedirs("../../../data/processed_data/model_data/single_rf_crf/model_with_errors/hyperparameter_configs/", exist_ok = True)

#Save the dictionary as a JSON file
file_path = f'../../../data/processed_data/model_data/single_rf_crf/model_with_errors/hyperparameter_configs/esm2_8m_nt_hyperparameters.json'
with open(file_path, 'w') as json_file:
    json.dump(model_config, json_file, indent=4)

print(f"Model configuration saved to {file_path}.")

