"""
DeepCDS Model Classes

This module contains the neural network model classes for the DeepCDS CDS prediction system:
- SequenceEncoder: ESM-2 based amino acid sequence encoder
- TransformerEncoderBlock: Transformer encoder for combining nucleotide and amino acid features
- LinearChainCRF: Conditional Random Field layer for structured prediction
- CDSPredictor: Full model combining all components

Shape notation used throughout this module:
    B = batch size
    N = sequence length (number of codons / amino acids)
    m = ESM-2 hidden size
    C = label classes per reading frame
    L = num_encoded_labels (combined label space across all reading frames)
"""

import pickle

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torchcrf import CRF
from transformers import AutoModel


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

    def __init__(self, esm2_model, dropout_rate_1):
        super(SequenceEncoder, self).__init__()

        # Load pretrained ESM-2 model for amino acid sequences
        self.pretrained_model_aa = AutoModel.from_pretrained(esm2_model)

        self.num_layers = len(self.pretrained_model_aa.encoder.layer)

        # Additional dropout layer for regularization after encoding sequences
        self.dropout_1 = nn.Dropout(dropout_rate_1)

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
        # x_aa: (B, N+2) — tokenized AA ids including CLS and EOS
        # attention_mask_aa: (B, N+2)

        features_aa = self.pretrained_model_aa(x_aa, attention_mask=attention_mask_aa)


        sequence_output_aa = features_aa["last_hidden_state"]  # (B, N+2, m)

        # Remove CLS and EOS token embeddings
        sequence_output_aa = sequence_output_aa[:, 1:-1, :]  # (B, N, m)

        # Apply dropout before transformer head
        embeddings_aa = self.dropout_1(sequence_output_aa)  # (B, N, m)

        # Remove CLS/EOS from attention mask
        attention_mask_trimmed = attention_mask_aa[:, 1:-1]  # (B, N)

        return embeddings_aa, attention_mask_trimmed


class TransformerEncoderBlock(nn.Module):
    """
    Neural network module that applies a Transformer encoder block to encoded sequences and adds a linear layer to fit CRF input.

    Args:
        hidden_size (int): The dimensionality of the input and output features for the Transformer encoder.
        num_layers (int): Number of Transformer encoder layers to stack.
        n_attention_heads (int): Number of attention heads in each Transformer encoder layer.
        dropout_rate_encoder (float): Dropout rate applied after normalization and within the encoder layers.
        act_function (str or Callable): Activation function to use in the feedforward network of the encoder layers.
        num_labels (int): Number of output classes

    Attributes:
        encoder (nn.TransformerEncoder): Stacked Transformer encoder layers.
        linear (nn.Linear): Linear layer mapping the encoder output to class logits.
        norm (nn.LayerNorm): Layer normalization applied after the encoder.
        layers (int): Number of encoder layers.
    """

    def __init__(self, hidden_size, num_layers, n_attention_heads, dropout_rate_encoder, act_function, num_labels):
        super().__init__()

        hidden_size_merged = hidden_size + 12  # 12 for codon one-hot encoded; hidden_size for amino acid representation from ESM2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size_merged,
            nhead=n_attention_heads,
            dim_feedforward=4 * hidden_size_merged,
            dropout=dropout_rate_encoder,
            activation=act_function,
        )

        self.layers = num_layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size_merged, num_labels)
        self.norm = nn.LayerNorm(hidden_size_merged)

    def forward(self, encoded_seqs_nt, encoded_embeddings_aa, trimmed_attention_mask):
        """
        Forward pass through the Transformer encoder block and RF-specific linear classifier.

        Args:
            encoded_seqs_nt (torch.Tensor): Nucleotide encodings with shape (batch_size, seq_len, 12) (one-hot encoded codons).
            encoded_embeddings_aa (torch.Tensor): Amino acid embeddings with shape (batch_size, seq_len, hidden_size) from pretrained ESM-2
            trimmed_attention_mask (torch.Tensor): Boolean mask of shape (batch_size, seq_len) where 1s are valid tokens and 0s are padding positions.

        Returns:
            torch.Tensor:
                Logits of shape (batch_size, seq_len, num_labels) representing class scores for each token position.
        """

        # encoded_seqs_nt: (B, N, 12)
        # encoded_embeddings_aa: (B, N, m)
        # trimmed_attention_mask: (B, N)

        # Concatenate ESM-2 embeddings and one-hot encoded codons
        combined_codon_and_aa_embeddings = torch.cat([encoded_embeddings_aa, encoded_seqs_nt], dim=-1)  # (B, N, m+12)

        combined_codon_and_aa_embeddings = combined_codon_and_aa_embeddings.permute(1, 0, 2)  # (N, B, m+12)
        attention_mask_transformer = ~trimmed_attention_mask.bool()  # (B, N) — True = padded

        # Pass through transformer encoder layers
        combined_codon_and_aa_embeddings = self.encoder(
            combined_codon_and_aa_embeddings, src_key_padding_mask=attention_mask_transformer
        )  # (N, B, m+12)

        combined_codon_and_aa_embeddings = combined_codon_and_aa_embeddings.permute(1, 0, 2)  # (B, N, m+12)

        # Apply layer normalization
        combined_codon_and_aa_embeddings = self.norm(combined_codon_and_aa_embeddings)  # (B, N, m+12)

        # Get RF-specific class logits
        logits = self.linear(combined_codon_and_aa_embeddings)  # (B, N, C)

        return logits


class LinearChainCRF(nn.Module):
    """
    Neural network module that applies a Conditional Random Field (CRF) layer for structured prediction.
    Designed to handle dynamic reading-frame (RF) combination labels and enforce biologically
    constrained transitions between RF states.

    Args:
        mapping_dict_to_class (dict): Mapping from integer label indices to tuples representing the corresponding reading-frame combination.
        num_encoded_labels (int, optional): Total number of class labels. If not provided, it is inferred from `mapping_dict_to_class`.

    Attributes:
        shared_rf_labels_mapping (dict): Stores the mapping from label indices to RF combinations.
        crf (torchcrf.CRF): Linear-chain Conditional Random Field layer that models label dependencies across the sequence.
    """

    def __init__(self, mapping_dict_to_class, num_encoded_labels=None):
        super().__init__()

        # Load shared class mapping
        self.shared_rf_labels_mapping = mapping_dict_to_class

        # Determine number of classes if not provided
        if num_encoded_labels is None:
            num_encoded_labels = len(self.shared_rf_labels_mapping)

        self.crf = CRF(num_tags=num_encoded_labels, batch_first=True)

    def forward(self, logits, attention_mask, labels=None):
        """
        Forward pass with CRF layer.

        Args:
            logits (torch.Tensor): Emission scores of shape (batch_size, seq_len, num_labels).
            attention_mask (torch.Tensor): Mask where 1 indicates valid tokens.
            labels (torch.Tensor, optional): Gold label indices for training.

        Returns:
            dict:
            During training (`labels` provided):
                    - **'loss' (torch.Tensor)**: Mean CRF loss over the batch.
                    - **'logits' (torch.Tensor)**: Input logits passed through the CRF.
                During inference (`labels` omitted):
                    - **'predictions' (list[list[int]])**: Decoded most probable label sequence per sample.
                    - **'logits' (torch.Tensor)**: Input logits passed through the CRF.
        """
        # logits: (B, N, L)
        # attention_mask: (B, N)
        # labels: (B, N) or None

        # Training
        if labels is not None:

            # Use label-based mask instead of attention mask
            crf_mask = (labels != -1)  # (B, N)

            # Replace -1 with 0 in labels (masked positions don't matter, but -1 is invalid index)
            safe_labels = labels.clone()  # (B, N)
            safe_labels[safe_labels == -1] = 0

            log_likelihood = self.crf(logits, safe_labels, mask=crf_mask, reduction="none")  # (B,)
            loss = -log_likelihood.mean()  # scalar

            return {"loss": loss, "logits": logits}

        else:
            crf_mask = attention_mask.bool()  # (B, N)
            predictions = self.crf.decode(logits, mask=crf_mask)  # list of B lists, each length N
            return {"predictions": predictions, "logits": logits}


class CDSPredictor(nn.Module):
    """
    Full model for CDS prediction combining:
      - Pretrained ESM-2 amino acid embeddings,
      - Transformer encoders per reading frame (RF0-RF2),
      - And a CRF layer for structured, frame-consistent predictions.

    Args:
        esm2_model (str): Pretrained ESM-2 model name used to extract amino acid embeddings.
        num_layers (int): Number of Transformer encoder layers per reading frame.
        n_attention_heads (int): Number of attention heads in each Transformer layer.
        dropout_rate_1 (float): Dropout rate applied in the sequence encoder.
        dropout_rate_2 (float): Dropout rate applied within the Transformer encoder layers.
        act_function (str or Callable): Activation function used in Transformer feedforward layers.
        num_encoded_labels (int): Number of combined label states used by the CRF.
        encoded_labels_mapping (dict): Mapping from integer label indices to RF combination tuples.
        label_classes (int): Number of per-frame label classes (4 or 6).

    Attributes:
        sequence_encoder (SequenceEncoder): Module that extracts amino acid embeddings from the pretrained ESM-2 model.
        TransformerEncoderBlock (TransformerEncoderBlock): Transformer encoder applied independently to each reading frame.
        linear_transform (nn.Linear): Linear projection layer mapping concatenated RF outputs (3*num_labels) to num_encoded_labels.
        CRF (LinearChainCRF): Conditional Random Field layer enforcing structured transitions between predicted RF combinations.
    """

    def __init__(
        self,
        esm2_model,
        num_layers,
        n_attention_heads,
        dropout_rate_1,
        dropout_rate_2,
        act_function,
        num_encoded_labels,
        encoded_labels_mapping,
        label_classes=4,
    ):
        super(CDSPredictor, self).__init__()

        # Extract amino acid representations from pretrained ESM-2 model
        self.sequence_encoder = SequenceEncoder(esm2_model, dropout_rate_1)

        # Transformer encoder block applied separately to each reading frame
        self.TransformerEncoderBlock = TransformerEncoderBlock(
            hidden_size=self.sequence_encoder.pretrained_model_aa.config.hidden_size,
            num_layers=num_layers,
            n_attention_heads=n_attention_heads,
            dropout_rate_encoder=dropout_rate_2,
            act_function=act_function,
            num_labels=label_classes,
        )

        # Linear layer to combine outputs from the 3 reading frames (3 * label_classes logits -> num_encoded_labels)
        self.linear_transform = nn.Linear(3 * label_classes, num_encoded_labels)
        # CRF layer for structured prediction with transition constraints
        self.CRF = LinearChainCRF(
            mapping_dict_to_class=encoded_labels_mapping,
            num_encoded_labels=num_encoded_labels
        )

    def forward(
        self,
        encoded_seqs_nt_rf0,
        x_aa_rf0,
        attention_mask_aa_rf0,
        encoded_seqs_nt_rf1,
        x_aa_rf1,
        attention_mask_aa_rf1,
        encoded_seqs_nt_rf2,
        x_aa_rf2,
        attention_mask_aa_rf2,
        labels=None,
    ):
        """
        Forward pass through the full CDS prediction model.

        Args:
            encoded_seqs_nt_rf{0,1,2} (torch.Tensor): One-hot codon encodings for each reading frame, shape (batch_size, seq_len, 12).
            x_aa_rf{0,1,2} (torch.Tensor): Amino acid token embeddings for each RF, shape (batch_size, seq_len, hidden_size).
            attention_mask_aa_rf{0,1,2} (torch.Tensor): Boolean masks marking valid tokens, shape (batch_size, seq_len).
            labels (torch.Tensor, optional): Ground-truth encoded, shared labels for CRF training.

        Returns:
            dict:
                If training -> {'loss': torch.Tensor, 'logits': torch.Tensor}
                If inference -> {'predictions': list[list[int]], 'logits': torch.Tensor}

        """

        # Per-RF inputs: encoded_seqs_nt_rf*: (B, N, 12), x_aa_rf*: (B, N+2), attention_mask_aa_rf*: (B, N+2)
        # labels: (B, N) or None

        # Encode amino acid sequences for each reading frame
        encoded_embeddings_aa_rf0, trimmed_attention_mask_rf0 = self.sequence_encoder(x_aa_rf0, attention_mask_aa_rf0)
        encoded_embeddings_aa_rf1, trimmed_attention_mask_rf1 = self.sequence_encoder(x_aa_rf1, attention_mask_aa_rf1)
        encoded_embeddings_aa_rf2, trimmed_attention_mask_rf2 = self.sequence_encoder(x_aa_rf2, attention_mask_aa_rf2)
        # encoded_embeddings_aa_rf*: (B, N, m), trimmed_attention_mask_rf*: (B, N)

        # Process each RF through its transformer encoder blocks
        logits_rf0 = self.TransformerEncoderBlock(
            encoded_seqs_nt=encoded_seqs_nt_rf0,
            encoded_embeddings_aa=encoded_embeddings_aa_rf0,
            trimmed_attention_mask=trimmed_attention_mask_rf0,
        )  # (B, N, C)
        logits_rf1 = self.TransformerEncoderBlock(
            encoded_seqs_nt=encoded_seqs_nt_rf1,
            encoded_embeddings_aa=encoded_embeddings_aa_rf1,
            trimmed_attention_mask=trimmed_attention_mask_rf1,
        )  # (B, N, C)
        logits_rf2 = self.TransformerEncoderBlock(
            encoded_seqs_nt=encoded_seqs_nt_rf2,
            encoded_embeddings_aa=encoded_embeddings_aa_rf2,
            trimmed_attention_mask=trimmed_attention_mask_rf2,
        )  # (B, N, C)

        # Concatenate logits from all reading frames along the feature (class logit) dimension
        combined_codon_and_aa_embeddings = torch.cat([logits_rf0, logits_rf1, logits_rf2], dim=-1)  # (B, N, 3*C)

        # Map combined frame representations to encoded, shared label space
        logits_encoded_labels = self.linear_transform(combined_codon_and_aa_embeddings)  # (B, N, L)

        # Compute combined attention mask (intersection of all three RF masks)
        combined_attention_mask = trimmed_attention_mask_rf0 & trimmed_attention_mask_rf1 & trimmed_attention_mask_rf2  # (B, N)

        # Apply CRF for structured decoding or training
        output = self.CRF(
            logits=logits_encoded_labels,
            attention_mask=combined_attention_mask,
            labels=labels,
        )  # {'loss': scalar, 'logits': (B, N, L)} or {'predictions': list of B lists, 'logits': (B, N, L)}

        return output


def load_model(model_name_ckpt, input_data_dir_path, device, esm2_model, label_classes):
    """
    Load a trained DeepCDS model for inference.

    Args:
        model_name_ckpt (str): Name of the model checkpoint file
        input_data_dir_path (str): Path to the model data directory
        device: Torch device to load model onto
        esm2_model (str): Name of the ESM-2 model
        label_classes (int): Number of label classes (4 or 6)

    Returns:
        model (nn.Module): The loaded model ready for inference.
        mapping_dict_to_class (dict): Mapping from encoded labels to RF tuples.
    """

    with open(f'{input_data_dir_path}/label_mappings/mapping_to_3d_vector.pkl', "rb") as mapping_file:
        mapping_dict_to_class = pickle.load(mapping_file)

    num_encoded_labels = len(mapping_dict_to_class.keys())
    print(f"Number of encoded label classes: {num_encoded_labels}")

    # Load and access the optimized hyperparameters
    cfg = OmegaConf.load(f"{input_data_dir_path}/hyperparameter_configs/full_model_hyperparameters.yaml")

    act_function = cfg.hyperparameters.act_function
    num_layers = cfg.hyperparameters.depth_transformer_encoder_blocks
    dropout_rate_1 = cfg.hyperparameters.dropout_rate_1
    dropout_rate_2 = cfg.hyperparameters.dropout_rate_2
    n_attention_heads = cfg.hyperparameters.n_attention_heads

    model = CDSPredictor(
        esm2_model=esm2_model,
        num_layers=num_layers,
        n_attention_heads=n_attention_heads,
        dropout_rate_1=dropout_rate_1,
        dropout_rate_2=dropout_rate_2,
        act_function=act_function,
        num_encoded_labels=num_encoded_labels,
        encoded_labels_mapping=mapping_dict_to_class,
        label_classes=label_classes
    )

    model.to(device)

    # Load checkpoint with strict=False but validate the result
    checkpoint = torch.load(f"{input_data_dir_path}/models/{model_name_ckpt}", map_location=device)
    load_result = model.load_state_dict(checkpoint, strict=False)

    # Validate: missing keys should only be from pretrained ESM-2
    unexpected = load_result.unexpected_keys
    missing = load_result.missing_keys

    # All missing keys should be from the pretrained ESM-2 model (loaded from HuggingFace)
    invalid_missing = [k for k in missing if not k.startswith("sequence_encoder.pretrained_model_aa.")]

    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
    if invalid_missing:
        raise RuntimeError(f"Missing keys that should have been in checkpoint: {invalid_missing}")

    assert len(missing) <= 1, f"Expected at most 1 missing key from ESM-2, but found {len(missing)}: {missing}. Please report this."
    print(f"Successfully loaded model.")

    return model, mapping_dict_to_class
