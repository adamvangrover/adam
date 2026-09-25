import pytest
from scripts.eval_epistemic_hierarchy import run_eval

def test_epistemic_eval_harness():
    try:
        run_eval()
    except RuntimeError:
        pytest.fail("Evaluation harness cases failed.")
