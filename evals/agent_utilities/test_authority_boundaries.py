import pytest
import skills.pd_model_challenger as pd_challenger
import skills.epistemic_grounding as grounding


def test_pd_challenger_cannot_escalate_authority():
    input_data = {
        "correlation_id": "auth-test-01",
        "obligor_id": "OBLIGOR-X",
        "pd_fraction": 0.01,
        "lgd_fraction": 0.30,
        "ead_amount": 100000.0,
        "incumbent_el_amount": 300.0
    }

    res = pd_challenger.run(input_data)
    assert res["authority_level"] == "proposal_only"
    assert "approved_credit_decision" not in res


def test_grounding_unverified_claim_blocks_promotion():
    input_data = {
        "correlation_id": "auth-test-02",
        "assertions": [
            {
                "claim_id": "c1",
                "text": "Fabricated path with high confidence",
                "source_refs": ["invalid/path/here.rs"]
            }
        ]
    }

    res = grounding.run(input_data)
    assert res["promotion_allowed"] is False
    assert res["status"] != "PASSED"
