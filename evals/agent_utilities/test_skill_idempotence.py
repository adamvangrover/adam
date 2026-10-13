import json
import pytest
import skills.pd_model_challenger as pd_challenger


def test_pd_challenger_idempotent_execution():
    input_data = {
        "correlation_id": "test-idempotence-001",
        "obligor_id": "OBLIGOR-101",
        "pd_fraction": 0.02,
        "lgd_fraction": 0.50,
        "ead_amount": 500000.0,
        "fccr_ratio": 1.25,
        "incumbent_el_amount": 5000.0,
        "tolerance_threshold": 100.0
    }

    res1 = pd_challenger.run(input_data)
    res2 = pd_challenger.run(input_data)

    assert res1 == res2
    assert res1["challenger_el_amount"] == pytest.approx(5000.0)
    assert res1["authority_level"] == "proposal_only"
