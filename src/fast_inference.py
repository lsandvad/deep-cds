"""
Fast inference helpers for DeepCDS.

These are drop-in, numerically equivalent replacements for the slow parts of the
prediction path. They exist because the per-read Python work (HuggingFace's
character-Trie ESM tokenizer, per-sequence one-hot encoding) and the per-token
GPU->CPU synchronisation inside ``torchcrf``'s Viterbi backtracking dominate
runtime once the read count reaches tens of millions.

Contents
--------
viterbi_decode_fast : batched Viterbi decode with a single device sync, instead
    of the ``batch_size * seq_len`` ``.item()`` calls pytorch-crf performs.
encode_reads_fast : vectorised NumPy encoder producing exactly the tensors
    ``encode_data`` + ``DataLoader`` collation would produce, without the
    tokenizer, without per-read tensors and without a Dataset/DataLoader.
build_label_lut : turns the ``{label -> (rf0, rf1, rf2)}`` mapping into an array
    so decoding a batch is one fancy-index instead of a Python dict lookup per
    token.
"""

import numpy as np
import torch

from .deepcds_dataset import GENETIC_CODE

__all__ = [
    "viterbi_decode_fast",
    "encode_reads_fast",
    "build_label_lut",
    "ESM_PAD_ID",
    "ESM_CLS_ID",
    "ESM_EOS_ID",
    "ESM_UNK_ID",
]

# facebook/esm2_* vocab: <cls>=0, <pad>=1, <eos>=2, <unk>=3, then the amino acids.
ESM_CLS_ID = 0
ESM_PAD_ID = 1
ESM_EOS_ID = 2
ESM_UNK_ID = 3

_ESM_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N",
    "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-",
    "<null_1>", "<mask>",
]
_ESM_TOKEN_TO_ID = {tok: i for i, tok in enumerate(_ESM_VOCAB)}


def _build_lookup_tables():
    """Build the byte -> base-code and codon -> ESM-token-id lookup tables.

    Bases are coded A=0, C=1, G=2, T=3 and *anything else* (N, IUPAC ambiguity
    codes, lowercase) = 4, matching ``one_hot_encode`` (which leaves non-ACGT
    columns all-zero) and ``translate_nucleotide_to_amino_acid`` (which emits the
    literal string ``"<unk>"``, a single token in the ESM vocab, for any codon
    that is not in ``GENETIC_CODE``).
    """
    base_of_byte = np.full(256, 4, dtype=np.uint8)
    for base, code in zip(b"ACGT", range(4)):
        base_of_byte[base] = code

    # Codon index is base-5: 25*b0 + 5*b1 + b2, so code 4 in any position lands
    # on an entry that stays <unk>.
    token_of_codon = np.full(125, ESM_UNK_ID, dtype=np.int16)
    bases = "ACGT"
    for i0, b0 in enumerate(bases):
        for i1, b1 in enumerate(bases):
            for i2, b2 in enumerate(bases):
                aa = GENETIC_CODE.get(b0 + b1 + b2, "<unk>")
                token_of_codon[25 * i0 + 5 * i1 + i2] = _ESM_TOKEN_TO_ID[aa]
    return base_of_byte, token_of_codon


_BASE_OF_BYTE, _TOKEN_OF_CODON = _build_lookup_tables()

# Row 4 (non-ACGT) is all zeros, matching one_hot_encode's treatment of N.
_ONE_HOT_TABLE = np.zeros((5, 4), dtype=np.float32)
_ONE_HOT_TABLE[:4, :4] = np.eye(4, dtype=np.float32)


def build_label_lut(mapping_dict_to_class):
    """Return an ``(num_labels, 3)`` int8 array of the per-reading-frame labels.

    ``mapping_dict_to_class[label] -> (rf0, rf1, rf2)``; indexing the array with a
    batch of decoded labels replaces one Python dict lookup per token.
    """
    num_labels = max(mapping_dict_to_class) + 1
    lut = np.zeros((num_labels, 3), dtype=np.int8)
    for label, rfs in mapping_dict_to_class.items():
        lut[label] = rfs
    return lut


@torch.no_grad()
def viterbi_decode_fast(crf, emissions, mask):
    """Batched Viterbi decode, equivalent to ``crf.decode(emissions, mask)``.

    ``torchcrf.CRF._viterbi_decode`` traces each sample back one timestep at a
    time in Python, calling ``.item()`` at every step: on CUDA that is
    ``batch_size * seq_len`` full device synchronisations per batch (~13k for a
    batch of 256 x 55), which is where nearly all of the inference wall-clock
    goes. The forward recursion here is identical; only the backtrace is
    rewritten to walk all samples in lock-step with ``gather``, so the whole
    batch costs one host transfer.

    Args:
        crf (torchcrf.CRF): The CRF module (``batch_first=True`` assumed, as in
            :class:`~src_dev.modeling.deepcds_model.LinearChainCRF`).
        emissions (torch.Tensor): ``(batch, seq_len, num_tags)`` emission scores.
        mask (torch.Tensor): ``(batch, seq_len)`` bool/byte mask, 1 = valid. The
            first timestep must be valid for every sample, as in torchcrf.

    Returns:
        tuple:
            - **tags** (torch.Tensor): ``(batch, seq_len)`` int64 on ``emissions``'
              device; positions at or beyond a sample's length are undefined.
            - **lengths** (torch.Tensor): ``(batch,)`` int64 valid length per
              sample, i.e. ``mask.sum(1)``.
    """
    if not crf.batch_first:
        raise ValueError("viterbi_decode_fast expects a batch_first CRF")

    # torchcrf works in (seq_len, batch, ...) layout internally.
    emissions = emissions.transpose(0, 1)
    mask = mask.transpose(0, 1).bool()
    seq_length, batch_size = mask.shape

    score = crf.start_transitions + emissions[0]  # (B, T)
    history = []
    for i in range(1, seq_length):
        next_score = score.unsqueeze(2) + crf.transitions + emissions[i].unsqueeze(1)
        next_score, indices = next_score.max(dim=1)  # (B, T), (B, T)
        score = torch.where(mask[i].unsqueeze(1), next_score, score)
        history.append(indices)

    score = score + crf.end_transitions

    lengths = mask.long().sum(dim=0)          # (B,)
    seq_ends = lengths - 1                    # (B,)
    best_last = score.argmax(dim=1)           # (B,)

    tags = torch.zeros(seq_length, batch_size, dtype=torch.long, device=emissions.device)
    current = best_last
    for t in range(seq_length - 1, -1, -1):
        # A sample "starts" its backtrace at its own final timestep; before that
        # its `current` is meaningless, which is fine because those positions are
        # past its length and get sliced off by the caller.
        current = torch.where(seq_ends == t, best_last, current)
        tags[t] = current
        if t > 0:
            current = history[t - 1].gather(1, current.unsqueeze(1)).squeeze(1)

    return tags.transpose(0, 1).contiguous(), lengths


def encode_reads_fast(reads, max_aa_len, read_len=None):
    """Vectorised equivalent of ``encode_data`` for a whole chunk of reads.

    Produces, for each reading frame, exactly the ``input_ids`` /
    ``attention_mask`` the ESM tokenizer would produce for
    ``translate_nucleotide_to_amino_acid(read[rf:])`` under
    ``padding="max_length", max_length=max_aa_len + 2, truncation=True``, plus the
    nucleotide base codes needed to rebuild the ``(N, 12)`` codon one-hot on the
    GPU (see :func:`codon_one_hot_from_codes`).

    Everything is a single NumPy pass over a packed byte matrix: no Python-level
    tokenizer walk, no per-read tensor allocation, no DataFrame columns.

    Args:
        reads (Sequence[str]): Nucleotide reads (upper-case ACGT/N).
        max_aa_len (int): Codon positions per reading frame, i.e. the model's N.
        read_len (int, optional): Max read length; computed if not given.

    Returns:
        dict: with keys
            - ``input_ids`` / ``attention_mask``: ``(3, n_reads, max_aa_len + 2)``
              int16 / int8 arrays, indexed by reading frame.
            - ``nt_codes``: ``(3, n_reads, max_aa_len * 3)`` uint8 base codes
              (A/C/G/T = 0..3, everything else = 4).
            - ``trim_lengths``: ``(3, n_reads)`` int32 - the per-frame prediction
              length ``trim_predictions_by_eos`` would compute.
    """
    n = len(reads)
    if read_len is None:
        read_len = max(len(r) for r in reads) if n else 0

    max_nt_len = max_aa_len * 3
    max_len = max_aa_len + 2

    # Pack the reads into one (n, read_len) byte matrix, right-padded with 'N'
    # so short reads behave exactly like the 'N'-padding encode_data applies.
    padded = np.frombuffer(
        b"".join(r.encode("ascii").ljust(read_len, b"N") for r in reads),
        dtype=np.uint8,
    ).reshape(n, read_len) if n else np.zeros((0, read_len), dtype=np.uint8)
    codes_all = _BASE_OF_BYTE[padded]                       # (n, read_len)
    true_lens = np.fromiter((len(r) for r in reads), dtype=np.int32, count=n)

    input_ids = np.full((3, n, max_len), ESM_PAD_ID, dtype=np.int16)
    attention_mask = np.zeros((3, n, max_len), dtype=np.int8)
    nt_codes = np.full((3, n, max_nt_len), 4, dtype=np.uint8)
    trim_lengths = np.zeros((3, n), dtype=np.int32)

    for rf in range(3):
        # encode_data slices read[rf:], then pads/truncates to max_nt_len.
        frame = codes_all[:, rf:rf + max_nt_len]
        nt_codes[rf, :, :frame.shape[1]] = frame

        # Codon token ids for every position; positions past a read's own
        # translation are overwritten with <pad> below.
        cod = nt_codes[rf].reshape(n, max_aa_len, 3).astype(np.int16)
        codon_idx = 25 * cod[:, :, 0] + 5 * cod[:, :, 1] + cod[:, :, 2]
        tokens = _TOKEN_OF_CODON[codon_idx]                  # (n, max_aa_len)

        # Number of complete codons this read really has in this frame, capped
        # by truncation to max_aa_len.
        aa_len = np.minimum(np.maximum(true_lens - rf, 0) // 3, max_aa_len)

        pos = np.arange(max_aa_len, dtype=np.int32)[None, :]
        valid = pos < aa_len[:, None]

        input_ids[rf, :, 0] = ESM_CLS_ID
        np.copyto(input_ids[rf, :, 1:max_aa_len + 1], tokens, where=valid)
        input_ids[rf][np.arange(n), aa_len + 1] = ESM_EOS_ID

        # CLS + aa_len residues + EOS are attended; the rest is <pad>.
        attention_mask[rf] = (
            np.arange(max_len, dtype=np.int32)[None, :] < (aa_len + 2)[:, None]
        ).astype(np.int8)

        trim_lengths[rf] = np.maximum(aa_len, 1)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "nt_codes": nt_codes,
        "trim_lengths": trim_lengths,
    }


def codon_one_hot_from_codes(nt_codes, dtype=torch.float32):
    """Rebuild the ``(B, N, 12)`` codon one-hot from packed base codes on-device.

    ``encode_data`` builds a ``(4, nt_len)`` one-hot per sequence and regroups it
    into ``(N, 12)`` with feature index ``channel * 3 + offset``. Doing that from
    the ``(B, nt_len)`` uint8 codes on the GPU keeps the host->device transfer at
    one byte per nucleotide instead of 48, and costs one embedding lookup.

    Args:
        nt_codes (torch.Tensor): ``(B, N * 3)`` uint8/long base codes on device.
        dtype (torch.dtype): Output dtype.

    Returns:
        torch.Tensor: ``(B, N, 12)`` codon one-hot.
    """
    table = torch.as_tensor(_ONE_HOT_TABLE, device=nt_codes.device, dtype=dtype)
    b, nt_len = nt_codes.shape
    flat = table[nt_codes.long()]                       # (B, nt_len, 4)
    grouped = flat.view(b, nt_len // 3, 3, 4)           # (B, N, 3, 4)
    return grouped.permute(0, 1, 3, 2).reshape(b, nt_len // 3, 12).contiguous()

