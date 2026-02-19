"""
Sliding Window Inference for DeepCDS and ESM2-only Models

This module provides sliding window inference for sequences longer than the
trained window size (300 nt / 100 codons). Instead of processing full-length
sequences that exceed the training distribution, it:
  1. Splits sequences into overlapping windows matching the training length
  2. Runs model inference on each window
  3. Averages pre-CRF logits in overlapping regions
  4. Runs a single CRF pass on the merged logit sequence

This keeps each model invocation within the trained distribution while
producing coherent full-length predictions.

Provides two entry points:
  - sliding_window_inference(): For the full DeepCDS model (nucleotide + amino acid)
  - sliding_window_inference_esm2(): For the ESM2-only model (amino acid only)
"""

import torch
import pandas as pd
from torch.utils.data import DataLoader

from .deepcds_dataset import encode_data

# Training configuration
TRAINED_WINDOW_SIZE_AA = 100  # max_aa_len used during training
DEFAULT_STRIDE_AA = 50        # stride in codons (50 codon overlap by default)


def get_window_positions(seq_len_nt, window_size_nt=300, stride_nt=210):
    """
    Compute codon-aligned sliding window start positions for a nucleotide sequence.

    All window starts are multiples of 3 to ensure codon boundary alignment
    across windows when merging logits. A right-aligned final window is added
    if regular striding doesn't cover the full sequence.

    Args:
        seq_len_nt (int): Length of the nucleotide sequence.
        window_size_nt (int): Window size in nucleotides (must be multiple of 3).
        stride_nt (int): Stride in nucleotides (must be multiple of 3).

    Returns:
        list[int]: Window start positions (each a multiple of 3).
    """
    assert stride_nt % 3 == 0, "Stride must be a multiple of 3 for codon alignment"
    assert window_size_nt % 3 == 0, "Window size must be a multiple of 3 for codon alignment"

    if seq_len_nt <= window_size_nt:
        return [0]

    positions = []
    start = 0
    while start + window_size_nt <= seq_len_nt:
        positions.append(start)
        start += stride_nt

    # Add right-aligned final window if the last regular window doesn't cover the end
    last_covered = positions[-1] + window_size_nt
    if last_covered < seq_len_nt:
        final_start = (seq_len_nt - window_size_nt) // 3 * 3
        if final_start > positions[-1]:
            positions.append(final_start)

    return positions


def _create_windowed_dataframe(chunk_df, window_starts, window_size_nt):
    """
    Expand a chunk of sequences into windowed subsequences.

    Each original sequence produces len(window_starts) rows, one per window.
    Window order is: all windows for sequence 0, then all for sequence 1, etc.

    Args:
        chunk_df (DataFrame): Original sequences with 'read', 'cds_coords',
            'indel_positions', 'read_name' columns.
        window_starts (list[int]): Window start positions from get_window_positions().
        window_size_nt (int): Window size in nucleotides.

    Returns:
        DataFrame: Windowed sequences with the same columns.
    """
    rows = []
    for _, row in chunk_df.iterrows():
        seq = row['read']
        for start in window_starts:
            window_seq = seq[start:start + window_size_nt]
            rows.append({
                'read': window_seq,
                'cds_coords': row['cds_coords'],
                'indel_positions': row['indel_positions'],
                'read_name': row['read_name'],
            })
    return pd.DataFrame(rows)


def _run_model_on_windows(model, window_dataset, total_windows, device, dtype,
                          batch_size, num_workers_cpu=0, pin_memory=False):
    """
    Run model forward pass on all windows and collect pre-CRF logits.

    Args:
        model: CDSPredictor model.
        window_dataset: SeqDataset of windowed sequences.
        total_windows (int): Total number of windows.
        device: Torch device.
        dtype: Torch dtype for float tensors (e.g., torch.float16).
        batch_size (int): Batch size for DataLoader.
        num_workers_cpu (int): DataLoader workers.
        pin_memory (bool): DataLoader pin_memory.

    Returns:
        torch.Tensor: Logits of shape (total_windows, window_size_aa, num_labels)
            in float32 for stable accumulation.
    """
    window_loader = DataLoader(
        window_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_cpu,
        pin_memory=pin_memory,
    )

    all_logits = []

    for batch in window_loader:
        nt_rf0 = batch['nt_encodings_rf0'].to(device, dtype=dtype)
        aa_rf0 = batch['aa_encodings_rf0']['input_ids'].to(device)
        mask_rf0 = batch['aa_encodings_rf0']['attention_mask'].to(device)

        nt_rf1 = batch['nt_encodings_rf1'].to(device, dtype=dtype)
        aa_rf1 = batch['aa_encodings_rf1']['input_ids'].to(device)
        mask_rf1 = batch['aa_encodings_rf1']['attention_mask'].to(device)

        nt_rf2 = batch['nt_encodings_rf2'].to(device, dtype=dtype)
        aa_rf2 = batch['aa_encodings_rf2']['input_ids'].to(device)
        mask_rf2 = batch['aa_encodings_rf2']['attention_mask'].to(device)

        outputs = model(
            nt_rf0, aa_rf0, mask_rf0,
            nt_rf1, aa_rf1, mask_rf1,
            nt_rf2, aa_rf2, mask_rf2,
        )

        # Extract pre-CRF logits and convert to FP32 for stable accumulation
        all_logits.append(outputs['logits'].float().detach())

        del nt_rf0, nt_rf1, nt_rf2
        del aa_rf0, aa_rf1, aa_rf2
        del mask_rf0, mask_rf1, mask_rf2
        del outputs

    return torch.cat(all_logits, dim=0)  # (total_windows, window_size_aa, L)


def _merge_window_logits(window_logits, window_starts, window_size_aa, full_aa_len,
                         num_labels, device):
    """
    Merge logits from overlapping windows by averaging.

    Args:
        window_logits (torch.Tensor): Shape (n_sequences, n_windows, window_size_aa, num_labels).
        window_starts (list[int]): Window start positions in nucleotides.
        window_size_aa (int): Window size in amino acids/codons.
        full_aa_len (int): Full sequence length in amino acids/codons.
        num_labels (int): Number of encoded label classes.
        device: Torch device.

    Returns:
        tuple: (merged_logits, merged_mask)
            merged_logits: shape (n_sequences, full_aa_len, num_labels) in float32
            merged_mask: shape (n_sequences, full_aa_len) boolean mask
    """
    n_sequences = window_logits.shape[0]

    merged_logits = torch.zeros(
        n_sequences, full_aa_len, num_labels, dtype=torch.float32, device=device
    )
    overlap_count = torch.zeros(
        n_sequences, full_aa_len, 1, dtype=torch.float32, device=device
    )

    for w_idx, start_nt in enumerate(window_starts):
        start_aa = start_nt // 3
        actual_len = min(window_size_aa, full_aa_len - start_aa)
        merged_logits[:, start_aa:start_aa + actual_len, :] += \
            window_logits[:, w_idx, :actual_len, :]
        overlap_count[:, start_aa:start_aa + actual_len, :] += 1

    # Average where covered
    merged_logits = merged_logits / overlap_count.clamp(min=1)

    # Mask: True where at least one window contributes
    merged_mask = (overlap_count.squeeze(-1) > 0)

    return merged_logits, merged_mask


def _decode_predictions(predictions_encoded, mapping_dict_to_class, seq_len):
    """
    Decode shared CRF predictions into per-reading-frame predictions.

    Args:
        predictions_encoded (list[list[int]]): CRF-decoded label indices per sequence.
        mapping_dict_to_class (dict): Mapping from encoded label index to (rf0, rf1, rf2) tuple.
        seq_len (int): Original nucleotide sequence length.

    Returns:
        tuple: (preds_rf0, preds_rf1, preds_rf2) — each a list of lists.
    """
    rf0_len = seq_len // 3
    rf1_len = (seq_len - 1) // 3
    rf2_len = (seq_len - 2) // 3

    preds_rf0, preds_rf1, preds_rf2 = [], [], []

    for preds_sample in predictions_encoded:
        decoded = [mapping_dict_to_class[p] for p in preds_sample]
        preds_rf0.append([rf[0] for rf in decoded][:rf0_len])
        preds_rf1.append([rf[1] for rf in decoded][:rf1_len])
        preds_rf2.append([rf[2] for rf in decoded][:rf2_len])

    return preds_rf0, preds_rf1, preds_rf2


def sliding_window_inference(model, sequences_df, seq_len, mapping_dict_to_class,
                             tokenizer, device, dtype, batch_size=256,
                             stride_aa=DEFAULT_STRIDE_AA,
                             num_workers_cpu=0, pin_memory=False):
    """
    Run sliding window inference on sequences longer than the trained window size.

    Splits each sequence into overlapping windows of TRAINED_WINDOW_SIZE_AA codons,
    runs the model on each window, averages logits in overlapping regions, and
    performs CRF decoding on the full merged logit sequence.

    Args:
        model (CDSPredictor): Trained DeepCDS model (in eval mode, on device).
        sequences_df (DataFrame): Input sequences with columns:
            'read', 'cds_coords', 'indel_positions', 'read_name'.
        seq_len (int): Nucleotide sequence length (same for all sequences in df).
        mapping_dict_to_class (dict): Mapping from encoded label index to RF tuples.
        tokenizer: ESM-2 tokenizer instance.
        device: Torch device.
        dtype: Torch dtype for float tensors (torch.float16 or torch.float32).
        batch_size (int): Base batch size (adjusted internally for windowing).
        stride_aa (int): Stride in amino acids/codons.
        num_workers_cpu (int): DataLoader workers.
        pin_memory (bool): DataLoader pin_memory.

    Yields:
        tuple: For each chunk of sequences:
            (preds_rf0, preds_rf1, preds_rf2, read_names, cds_coords, seq_errors, chunk_size)
    """
    window_size_aa = TRAINED_WINDOW_SIZE_AA
    window_size_nt = window_size_aa * 3
    stride_nt = stride_aa * 3

    window_starts = get_window_positions(seq_len, window_size_nt, stride_nt)
    n_windows = len(window_starts)
    full_aa_len = seq_len // 3
    num_labels = model.linear_transform.out_features

    # Adjust batch size: each original sequence becomes n_windows windows
    effective_batch_size = max(1, batch_size // n_windows)
    # Batch size for model forward pass on windows
    window_batch_size = batch_size

    n_sequences = len(sequences_df)

    print(f"  Sliding window: {n_windows} windows per sequence "
          f"(size={window_size_nt}nt, stride={stride_nt}nt, overlap={window_size_nt - stride_nt}nt)")
    print(f"  Effective batch size: {effective_batch_size} sequences "
          f"({effective_batch_size * n_windows} windows)")

    for chunk_start in range(0, n_sequences, effective_batch_size):
        chunk_df = sequences_df.iloc[chunk_start:chunk_start + effective_batch_size].reset_index(drop=True)
        chunk_size = len(chunk_df)

        # Expand sequences into windows
        windowed_df = _create_windowed_dataframe(chunk_df, window_starts, window_size_nt)

        # Encode all windows with the trained window size
        window_dataset = encode_data(windowed_df, window_size_aa, tokenizer)

        total_windows = chunk_size * n_windows

        # Run model on all windows
        all_logits = _run_model_on_windows(
            model, window_dataset, total_windows, device, dtype,
            batch_size=window_batch_size,
            num_workers_cpu=num_workers_cpu,
            pin_memory=pin_memory,
        )

        # Reshape: (chunk_size, n_windows, window_size_aa, num_labels)
        all_logits = all_logits.view(chunk_size, n_windows, window_size_aa, num_labels)

        # Merge overlapping windows
        merged_logits, merged_mask = _merge_window_logits(
            all_logits, window_starts, window_size_aa, full_aa_len, num_labels, device
        )

        # Convert to model dtype for CRF compatibility
        if dtype == torch.float16:
            merged_logits = merged_logits.half()
        merged_mask = merged_mask.bool()

        # CRF decoding on merged full-length logits
        predictions_encoded = model.CRF.crf.decode(merged_logits, mask=merged_mask)

        # Decode shared predictions to per-RF predictions
        preds_rf0, preds_rf1, preds_rf2 = _decode_predictions(
            predictions_encoded, mapping_dict_to_class, seq_len
        )

        # Extract metadata
        read_names = chunk_df['read_name'].tolist()
        cds_coords = chunk_df['cds_coords'].tolist()
        seq_errors = chunk_df['indel_positions'].astype(str).tolist()

        yield preds_rf0, preds_rf1, preds_rf2, read_names, cds_coords, seq_errors, chunk_size

        # Cleanup
        del all_logits, merged_logits, merged_mask, windowed_df, window_dataset
        del predictions_encoded


def _run_model_on_windows_esm2(model, window_dataset, total_windows, device, dtype,
                                batch_size, num_workers_cpu=0, pin_memory=False):
    """
    Run ESM2-only model forward pass on all windows and collect pre-CRF logits.

    Same as _run_model_on_windows but for the ESM2-only model which takes
    only amino acid encodings (no nucleotide inputs).

    Args:
        model: CDSPredictorESM2 model.
        window_dataset: SeqDatasetESM2 of windowed sequences.
        total_windows (int): Total number of windows.
        device: Torch device.
        dtype: Torch dtype for float tensors.
        batch_size (int): Batch size for DataLoader.
        num_workers_cpu (int): DataLoader workers.
        pin_memory (bool): DataLoader pin_memory.

    Returns:
        torch.Tensor: Logits of shape (total_windows, window_size_aa, num_labels)
            in float32 for stable accumulation.
    """
    window_loader = DataLoader(
        window_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_cpu,
        pin_memory=pin_memory,
    )

    all_logits = []

    for batch in window_loader:
        aa_rf0 = batch['aa_encodings_rf0']['input_ids'].to(device)
        mask_rf0 = batch['aa_encodings_rf0']['attention_mask'].to(device)

        aa_rf1 = batch['aa_encodings_rf1']['input_ids'].to(device)
        mask_rf1 = batch['aa_encodings_rf1']['attention_mask'].to(device)

        aa_rf2 = batch['aa_encodings_rf2']['input_ids'].to(device)
        mask_rf2 = batch['aa_encodings_rf2']['attention_mask'].to(device)

        outputs = model(
            aa_rf0, mask_rf0,
            aa_rf1, mask_rf1,
            aa_rf2, mask_rf2,
        )

        # Extract pre-CRF logits and convert to FP32 for stable accumulation
        all_logits.append(outputs['logits'].float().detach())

        del aa_rf0, aa_rf1, aa_rf2
        del mask_rf0, mask_rf1, mask_rf2
        del outputs

    return torch.cat(all_logits, dim=0)  # (total_windows, window_size_aa, L)


def sliding_window_inference_esm2(model, sequences_df, seq_len, mapping_dict_to_class,
                                   encode_fn, tokenizer, device, dtype, batch_size=256,
                                   stride_aa=DEFAULT_STRIDE_AA,
                                   num_workers_cpu=0, pin_memory=False):
    """
    Run sliding window inference for the ESM2-only model on long sequences.

    Same approach as sliding_window_inference but for the ESM2-only model:
    no nucleotide encoding, different model forward signature.

    Args:
        model (CDSPredictorESM2): Trained ESM2-only model (in eval mode, on device).
        sequences_df (DataFrame): Input sequences with columns:
            'read', 'cds_coords', 'indel_positions', 'read_name'.
        seq_len (int): Nucleotide sequence length (same for all sequences in df).
        mapping_dict_to_class (dict): Mapping from encoded label index to RF tuples.
        encode_fn (callable): Encoding function with signature
            encode_fn(df, max_aa_len, tokenizer) -> Dataset.
        tokenizer: ESM-2 tokenizer instance.
        device: Torch device.
        dtype: Torch dtype for float tensors (torch.float16 or torch.float32).
        batch_size (int): Base batch size (adjusted internally for windowing).
        stride_aa (int): Stride in amino acids/codons.
        num_workers_cpu (int): DataLoader workers.
        pin_memory (bool): DataLoader pin_memory.

    Yields:
        tuple: For each chunk of sequences:
            (preds_rf0, preds_rf1, preds_rf2, read_names, cds_coords, seq_errors, chunk_size)
    """
    window_size_aa = TRAINED_WINDOW_SIZE_AA
    window_size_nt = window_size_aa * 3
    stride_nt = stride_aa * 3

    window_starts = get_window_positions(seq_len, window_size_nt, stride_nt)
    n_windows = len(window_starts)
    full_aa_len = seq_len // 3
    num_labels = model.linear_transform.out_features

    # Adjust batch size: each original sequence becomes n_windows windows
    effective_batch_size = max(1, batch_size // n_windows)
    # Batch size for model forward pass on windows
    window_batch_size = batch_size

    n_sequences = len(sequences_df)

    print(f"  Sliding window: {n_windows} windows per sequence "
          f"(size={window_size_nt}nt, stride={stride_nt}nt, overlap={window_size_nt - stride_nt}nt)")
    print(f"  Effective batch size: {effective_batch_size} sequences "
          f"({effective_batch_size * n_windows} windows)")

    for chunk_start in range(0, n_sequences, effective_batch_size):
        chunk_df = sequences_df.iloc[chunk_start:chunk_start + effective_batch_size].reset_index(drop=True)
        chunk_size = len(chunk_df)

        # Expand sequences into windows
        windowed_df = _create_windowed_dataframe(chunk_df, window_starts, window_size_nt)

        # Encode all windows with the trained window size
        window_dataset = encode_fn(windowed_df, window_size_aa, tokenizer)

        total_windows = chunk_size * n_windows

        # Run ESM2 model on all windows
        all_logits = _run_model_on_windows_esm2(
            model, window_dataset, total_windows, device, dtype,
            batch_size=window_batch_size,
            num_workers_cpu=num_workers_cpu,
            pin_memory=pin_memory,
        )

        # Reshape: (chunk_size, n_windows, window_size_aa, num_labels)
        all_logits = all_logits.view(chunk_size, n_windows, window_size_aa, num_labels)

        # Merge overlapping windows
        merged_logits, merged_mask = _merge_window_logits(
            all_logits, window_starts, window_size_aa, full_aa_len, num_labels, device
        )

        # Convert to model dtype for CRF compatibility
        if dtype == torch.float16:
            merged_logits = merged_logits.half()
        merged_mask = merged_mask.bool()

        # CRF decoding on merged full-length logits
        predictions_encoded = model.CRF.crf.decode(merged_logits, mask=merged_mask)

        # Decode shared predictions to per-RF predictions
        preds_rf0, preds_rf1, preds_rf2 = _decode_predictions(
            predictions_encoded, mapping_dict_to_class, seq_len
        )

        # Extract metadata
        read_names = chunk_df['read_name'].tolist()
        cds_coords = chunk_df['cds_coords'].tolist()
        seq_errors = chunk_df['indel_positions'].astype(str).tolist()

        yield preds_rf0, preds_rf1, preds_rf2, read_names, cds_coords, seq_errors, chunk_size

        # Cleanup
        del all_logits, merged_logits, merged_mask, windowed_df, window_dataset
        del predictions_encoded
