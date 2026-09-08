# DeepCDS modeling module
from .deepcds_model import (
    CDSPredictor,
    LinearChainCRF,
    SequenceEncoder,
    TransformerEncoderBlock,
    load_model,
)
from .deepcds_dataset import (
    GENETIC_CODE,
    SeqDataset,
    encode_data,
    one_hot_encode,
    process_nt_sequences_to_codons,
    translate_nucleotide_to_amino_acid,
)
from .fast_inference import (
    build_label_lut,
    codon_one_hot_from_codes,
    encode_reads_fast,
    viterbi_decode_fast,
)
from .sliding_window import (
    TRAINED_WINDOW_SIZE_AA,
    DEFAULT_STRIDE_AA,
    get_window_positions,
    sliding_window_inference,
    sliding_window_inference_esm2,
)

__all__ = [
    # Model classes
    "CDSPredictor",
    "LinearChainCRF",
    "SequenceEncoder",
    "TransformerEncoderBlock",
    "load_model",
    # Dataset classes and functions
    "GENETIC_CODE",
    "SeqDataset",
    "encode_data",
    "one_hot_encode",
    "process_nt_sequences_to_codons",
    "translate_nucleotide_to_amino_acid",
    # Fast inference helpers
    "build_label_lut",
    "codon_one_hot_from_codes",
    "encode_reads_fast",
    "viterbi_decode_fast",
    # Sliding window inference
    "TRAINED_WINDOW_SIZE_AA",
    "DEFAULT_STRIDE_AA",
    "get_window_positions",
    "sliding_window_inference",
    "sliding_window_inference_esm2",
]


