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
]
