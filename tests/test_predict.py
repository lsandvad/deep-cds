"""
Unit tests for the DeepCDS prediction pipeline.

Model-dependent tests are skipped automatically when checkpoint files are not
present (e.g. in CI without model artifacts). Run locally after copying the
.pth checkpoint into models/none/.
"""

import os
import subprocess
import sys
import tempfile

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Paths under the production layout
_MODEL_CKPT_NAME = "full_model_all_genomes_seed_42_trained_final.pth"
_CKPT_PATH = os.path.join(ROOT_DIR, "models", "model_without_errors", _MODEL_CKPT_NAME)
_LABEL_MAPPING_PATH = os.path.join(ROOT_DIR, "configs", "model_without_errors", "label_mapping.pkl")
_HYPERPARAMS_PATH = os.path.join(ROOT_DIR, "configs", "model_without_errors", "hyperparameters.yaml")
_TEST_FASTA = os.path.join(ROOT_DIR, "test.fasta")

requires_model = pytest.mark.skipif(
    not os.path.isfile(_CKPT_PATH),
    reason="Model checkpoint not found in models/model_without_errors/ — copy the .pth file there to run this test",
)


@requires_model
def test_model_loads_correctly():
    """Model loads without unexpected keys; missing keys must be ≤1 (ESM-2 pooler only).

    The assertion `assert len(missing) <= 1` is enforced inside load_model itself;
    reaching here without an AssertionError confirms it passed.
    """
    import torch
    from src import load_model

    model, mapping = load_model(
        ckpt_path=_CKPT_PATH,
        label_mapping_path=_LABEL_MAPPING_PATH,
        hyperparams_path=_HYPERPARAMS_PATH,
        device=torch.device("cpu"),
        esm2_model="facebook/esm2_t6_8M_UR50D",
        label_classes=4,
    )
    assert model is not None
    assert len(mapping) > 0


@requires_model
def test_prediction_runs_on_test_fasta():
    """predict.py produces a valid GFF3 file for the bundled test.fasta."""
    assert os.path.isfile(_TEST_FASTA), f"test.fasta not found at {_TEST_FASTA}"

    with tempfile.NamedTemporaryFile(suffix=".gff", delete=False) as tmp:
        output_path = tmp.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(ROOT_DIR, "predict.py"),
                "--fasta", _TEST_FASTA,
                "--error_type", "none",
                "--output", output_path,
            ],
            capture_output=True,
            text=True,
            cwd=ROOT_DIR,
        )
        assert result.returncode == 0, (
            f"predict.py exited with code {result.returncode}.\n"
            f"stderr:\n{result.stderr}"
        )
        assert os.path.isfile(output_path), "Output GFF file was not created"
        with open(output_path) as f:
            content = f.read()
        assert content.startswith("##gff-version 3"), "GFF output missing version header"
    finally:
        if os.path.isfile(output_path):
            os.remove(output_path)
