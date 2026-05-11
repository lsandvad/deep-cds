"""
Tests for the DeepCDS prediction pipeline.

Unit tests (no model) always run. End-to-end CLI tests require the model
checkpoints in models/; they are skipped automatically when absent.
"""

import gzip
import os
import subprocess
import sys

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

SCRIPT = os.path.join(ROOT_DIR, "predict_with_deepcds.py")
FASTA_PLAIN = os.path.join(ROOT_DIR, "data_example", "test.fasta")
FASTA_LARGE = os.path.join(ROOT_DIR, "data_example", "GCF_000007365.1_test.fasta")
GFF_EXAMPLE = os.path.join(ROOT_DIR, "data_example", "GCF_000007365.1_test_deepcds_predictions.gff")
MODEL_NONE = os.path.join(ROOT_DIR, "models", "deepcds.pth")
MODEL_S = os.path.join(ROOT_DIR, "models", "deepcds_S.pth")
MODEL_SI = os.path.join(ROOT_DIR, "models", "deepcds_SI.pth")

requires_models = pytest.mark.skipif(
    not all(os.path.isfile(p) for p in [MODEL_NONE, MODEL_S, MODEL_SI]),
    reason="Model checkpoints not found in models/ — skipping end-to-end tests",
)


def _run(args, cwd=None):
    return subprocess.run(
        [sys.executable, SCRIPT] + args,
        capture_output=True,
        text=True,
        cwd=cwd or ROOT_DIR,
    )


# ── Unit tests: postprocessing ────────────────────────────────────────────────

def test_translate_cds_standard_codons():
    from src.postprocessing import translate_cds
    assert translate_cds("ATGGCC") == "MA"
    assert translate_cds("ATGTAA") == "M*"


def test_translate_cds_ambiguous_n_codon():
    from src.postprocessing import translate_cds
    assert translate_cds("ATGNCC") == "MX"


def test_translate_cds_incomplete_terminal_codon_ignored():
    from src.postprocessing import translate_cds
    assert translate_cds("ATGGCCA") == "MA"


def test_reverse_complement():
    from src.postprocessing import reverse_complement
    assert reverse_complement("ATGC") == "GCAT"
    assert reverse_complement("AAAA") == "TTTT"
    assert reverse_complement("AACCGGTT") == "AACCGGTT"


def test_extract_cds_from_gff_plain_fasta(tmp_path):
    from src.postprocessing import extract_cds_from_gff
    fna = str(tmp_path / "out.fna")
    faa = str(tmp_path / "out.faa")
    extract_cds_from_gff(FASTA_LARGE, GFF_EXAMPLE, fna, faa)
    assert os.path.getsize(fna) > 0
    assert os.path.getsize(faa) > 0


def test_extract_cds_from_gff_gzipped_fasta(tmp_path):
    from src.postprocessing import extract_cds_from_gff
    gz_fasta = str(tmp_path / "input.fasta.gz")
    fna = str(tmp_path / "out.fna")
    faa = str(tmp_path / "out.faa")
    with open(FASTA_LARGE, "rb") as src, gzip.open(gz_fasta, "wb") as dst:
        dst.write(src.read())
    extract_cds_from_gff(gz_fasta, GFF_EXAMPLE, fna, faa)
    assert os.path.getsize(fna) > 0
    assert os.path.getsize(faa) > 0


def test_extract_cds_from_gff_plain_and_gz_same_output(tmp_path):
    from src.postprocessing import extract_cds_from_gff
    gz_fasta = str(tmp_path / "input.fasta.gz")
    with open(FASTA_LARGE, "rb") as src, gzip.open(gz_fasta, "wb") as dst:
        dst.write(src.read())
    fna_plain = str(tmp_path / "plain.fna")
    fna_gz = str(tmp_path / "gz.fna")
    extract_cds_from_gff(FASTA_LARGE, GFF_EXAMPLE, fna_plain, None)
    extract_cds_from_gff(gz_fasta, GFF_EXAMPLE, fna_gz, None)
    with open(fna_plain) as f:
        content_plain = f.read()
    with open(fna_gz) as f:
        content_gz = f.read()
    assert content_plain == content_gz


# ── Unit tests: FASTA parsing ─────────────────────────────────────────────────

def test_parse_fasta_plain_returns_sequences():
    from predict_with_deepcds import parse_fasta
    records = parse_fasta(FASTA_PLAIN)
    assert len(records) > 0
    for name, seq in records:
        assert isinstance(name, str) and len(name) > 0
        assert isinstance(seq, str) and len(seq) > 0


def test_parse_fasta_gzipped_matches_plain(tmp_path):
    from predict_with_deepcds import parse_fasta
    gz_fasta = str(tmp_path / "test.fasta.gz")
    with open(FASTA_PLAIN, "rb") as src, gzip.open(gz_fasta, "wb") as dst:
        dst.write(src.read())
    records_plain = parse_fasta(FASTA_PLAIN)
    records_gz = parse_fasta(gz_fasta)
    assert records_plain == records_gz


def test_parse_fasta_sequences_are_uppercase():
    from predict_with_deepcds import parse_fasta
    records = parse_fasta(FASTA_PLAIN)
    for _, seq in records:
        assert seq == seq.upper()


# ── CLI behavior tests (no model needed) ─────────────────────────────────────

def test_cli_help_exits_zero():
    result = _run(["--help"])
    assert result.returncode == 0
    assert "input_fasta" in result.stdout


def test_cli_no_args_exits_nonzero():
    result = _run([])
    assert result.returncode != 0


def test_cli_invalid_error_model():
    result = _run(["-in", FASTA_PLAIN, "--error_model", "invalid"])
    assert result.returncode != 0


def test_cli_invalid_suppress_output_files():
    result = _run(["-in", FASTA_PLAIN, "--error_model", "none",
                   "--suppress_output_files", "xyz"])
    assert result.returncode != 0


# ── End-to-end tests (require model checkpoints) ─────────────────────────────

@requires_models
def test_e2e_error_model_none_plain_fasta(tmp_path):
    out = str(tmp_path / "out")
    result = _run(["-in", FASTA_PLAIN, "--error_model", "none",
                   "--output", out, "--compute_device", "cpu"])
    assert result.returncode == 0, result.stderr
    assert os.path.isfile(out + ".gff")
    assert os.path.isfile(out + ".fna")
    assert os.path.isfile(out + ".faa")


@requires_models
def test_e2e_error_model_S(tmp_path):
    out = str(tmp_path / "out")
    result = _run(["-in", FASTA_PLAIN, "--error_model", "S",
                   "--output", out, "--compute_device", "cpu"])
    assert result.returncode == 0, result.stderr
    assert os.path.isfile(out + ".gff")


@requires_models
def test_e2e_error_model_SI(tmp_path):
    out = str(tmp_path / "out")
    result = _run(["-in", FASTA_PLAIN, "--error_model", "SI",
                   "--output", out, "--compute_device", "cpu"])
    assert result.returncode == 0, result.stderr
    assert os.path.isfile(out + ".gff")


@requires_models
def test_e2e_gzipped_fasta_input(tmp_path):
    gz_fasta = str(tmp_path / "test.fasta.gz")
    with open(FASTA_PLAIN, "rb") as src, gzip.open(gz_fasta, "wb") as dst:
        dst.write(src.read())
    out = str(tmp_path / "out")
    result = _run(["-in", gz_fasta, "--error_model", "none",
                   "--output", out, "--compute_device", "cpu"])
    assert result.returncode == 0, result.stderr
    assert os.path.isfile(out + ".gff")
    assert os.path.isfile(out + ".fna")
    assert os.path.isfile(out + ".faa")


@requires_models
def test_e2e_output_stem_strips_all_extensions(tmp_path):
    """Output stem for myreads.fasta.gz should be myreads, not myreads.fasta."""
    gz_fasta = str(tmp_path / "myreads.fasta.gz")
    with open(FASTA_PLAIN, "rb") as src, gzip.open(gz_fasta, "wb") as dst:
        dst.write(src.read())
    result = _run(["-in", gz_fasta, "--error_model", "none", "--compute_device", "cpu"],
                  cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    files = os.listdir(tmp_path)
    assert any(f.startswith("myreads_deepcds") for f in files), files
    assert not any(".fasta_deepcds" in f for f in files), files


@requires_models
def test_e2e_suppress_fna_faa(tmp_path):
    out = str(tmp_path / "out")
    result = _run(["-in", FASTA_PLAIN, "--error_model", "none",
                   "--output", out, "--compute_device", "cpu",
                   "--suppress_output_files", "fna,faa"])
    assert result.returncode == 0, result.stderr
    assert os.path.isfile(out + ".gff")
    assert not os.path.isfile(out + ".fna")
    assert not os.path.isfile(out + ".faa")


@requires_models
def test_e2e_suppress_gff(tmp_path):
    out = str(tmp_path / "out")
    result = _run(["-in", FASTA_PLAIN, "--error_model", "none",
                   "--output", out, "--compute_device", "cpu",
                   "--suppress_output_files", "gff"])
    assert result.returncode == 0, result.stderr
    assert not os.path.isfile(out + ".gff")
    assert os.path.isfile(out + ".fna")
    assert os.path.isfile(out + ".faa")


@requires_models
def test_e2e_gzip_output(tmp_path):
    out = str(tmp_path / "out")
    result = _run(["-in", FASTA_PLAIN, "--error_model", "none",
                   "--output", out, "--compute_device", "cpu",
                   "--gzip_output"])
    assert result.returncode == 0, result.stderr
    for ext in [".gff.gz", ".fna.gz", ".faa.gz"]:
        assert os.path.isfile(out + ext), f"Missing {ext}"
    with gzip.open(out + ".gff.gz", "rt") as f:
        assert f.readline().startswith("##gff-version 3")


@requires_models
def test_e2e_gff_has_valid_format(tmp_path):
    out = str(tmp_path / "out")
    result = _run(["-in", FASTA_PLAIN, "--error_model", "none",
                   "--output", out, "--compute_device", "cpu"])
    assert result.returncode == 0, result.stderr
    with open(out + ".gff") as f:
        lines = f.readlines()
    assert lines[0].startswith("##gff-version 3")
    cds_lines = [l for l in lines if not l.startswith("#") and "\tCDS\t" in l]
    assert len(cds_lines) > 0
    for line in cds_lines:
        fields = line.strip().split("\t")
        assert len(fields) == 9
        assert int(fields[3]) <= int(fields[4])


@requires_models
def test_e2e_min_cds_length_filters_short(tmp_path):
    out_default = str(tmp_path / "default")
    out_strict = str(tmp_path / "strict")
    _run(["-in", FASTA_PLAIN, "--error_model", "none",
          "--output", out_default, "--compute_device", "cpu"])
    _run(["-in", FASTA_PLAIN, "--error_model", "none",
          "--output", out_strict, "--compute_device", "cpu",
          "--min_cds_length", "300"])
    with open(out_default + ".gff") as f:
        n_default = sum(1 for l in f if "\tCDS\t" in l)
    with open(out_strict + ".gff") as f:
        n_strict = sum(1 for l in f if "\tCDS\t" in l)
    assert n_strict <= n_default


@requires_models
def test_e2e_fna_sequences_are_nucleotides(tmp_path):
    out = str(tmp_path / "out")
    _run(["-in", FASTA_PLAIN, "--error_model", "none",
          "--output", out, "--compute_device", "cpu"])
    with open(out + ".fna") as f:
        for line in f:
            if not line.startswith(">"):
                assert set(line.strip().upper()).issubset(set("ACGTN"))


@requires_models
def test_e2e_faa_sequences_are_amino_acids(tmp_path):
    out = str(tmp_path / "out")
    _run(["-in", FASTA_PLAIN, "--error_model", "none",
          "--output", out, "--compute_device", "cpu"])
    valid_aa = set("ACDEFGHIKLMNPQRSTVWYX*")
    with open(out + ".faa") as f:
        for line in f:
            if not line.startswith(">"):
                assert set(line.strip().upper()).issubset(valid_aa)
