import os
import json
import pytest
from skills.llm_as_judge_arbiter import run as run_judge_arbiter


def test_llm_as_judge_arbiter_single_candidate_pass():
    input_data = {
        "correlation_id": "test-corr-001",
        "evaluation_type": "single_candidate_rubric",
        "candidate_proposal": {
            "id": "PROP-2026-001",
            "proposal_type": "credit_memo",
            "financial_metrics": {
                "dscr": 1.45,
                "leverage_ratio": 3.5,
                "unauthorized_addbacks": False,
                "refinancing_wall_modeled": True,
                "margin_compression_modeled": True,
                "non_linear_debt_service_decay": True,
                "sr11_7_compliant": True,
                "snc_classification": "PASS"
            },
            "evidence_mappings": [
                {"assertion_id": "A1", "text": "Q3 EBITDA was $50M", "source_ref": "sec_filing_10q.pdf"},
                {"assertion_id": "A2", "text": "Total Debt is $175M", "source_ref": "sec_filing_10k.pdf"}
            ]
        },
        "tripwire_policy": {
            "min_dscr": 1.05,
            "allow_unauthorized_addbacks": False,
            "require_sr11_7": True
        }
    }

    result = run_judge_arbiter(input_data)
    assert result["skill_id"] == "llm_as_judge_arbiter"
    assert result["status"] == "PASSED"
    assert result["outcome"] == "ADMIT_PROPOSAL"
    assert result["tripwire_status"]["passed"] is True
    assert result["rubric_scores"]["weighted_composite_score"] >= 0.85


def test_llm_as_judge_arbiter_tripwire_hard_reject():
    input_data = {
        "correlation_id": "test-corr-002",
        "evaluation_type": "single_candidate_rubric",
        "candidate_proposal": {
            "id": "PROP-2026-002",
            "financial_metrics": {
                "dscr": 0.98,  # DSCR breach (< 1.05)
                "unauthorized_addbacks": True,  # Unauthorized covenant addbacks
                "refinancing_wall_modeled": True,
                "margin_compression_modeled": True,
                "non_linear_debt_service_decay": True,
                "sr11_7_compliant": True
            },
            "evidence_mappings": [
                {"assertion_id": "A1", "text": "10-K filing reference", "source_ref": "sec_10k.pdf"}
            ]
        },
        "tripwire_policy": {
            "min_dscr": 1.05,
            "allow_unauthorized_addbacks": False
        }
    }

    result = run_judge_arbiter(input_data)
    assert result["status"] == "REJECTED"
    assert result["outcome"] == "REJECT"
    assert result["tripwire_status"]["passed"] is False
    assert any("TRIPWIRE_DSCR_BREACH" in t for t in result["tripwire_status"]["triggered_tripwires"])
    assert any("TRIPWIRE_UNAUTHORIZED_COVENANT_ADDBACKS" in t for t in result["tripwire_status"]["triggered_tripwires"])


def test_llm_as_judge_arbiter_pairwise_debate_bias_mitigation():
    input_data = {
        "correlation_id": "test-corr-003",
        "evaluation_type": "pairwise_debate",
        "candidate_debate": {
            "candidate_a": {
                "agent_id": "underwriting_agent_v1",
                "proposal_text": "Extremely long verbose proposal text with lots of fluff...",
                "token_count": 2500,
                "grounding_count": 5
            },
            "candidate_b": {
                "agent_id": "challenger_agent_v2",
                "proposal_text": "Concise high-density grounded proposal",
                "token_count": 400,
                "grounding_count": 8
            }
        }
    }

    result = run_judge_arbiter(input_data)
    assert result["pairwise_adjudication"] is not None
    adj = result["pairwise_adjudication"]
    assert adj["winner"] == "candidate_b"
    assert adj["bias_mitigation_applied"]["position_swapped"] is True
    assert adj["bias_mitigation_applied"]["verbosity_length_normalized"] is True


def test_llm_as_judge_arbiter_human_calibration():
    input_data = {
        "correlation_id": "test-corr-004",
        "evaluation_type": "calibration_tune",
        "calibration_dataset": [
            {"proposal_id": "P1", "judge_verdict": "ADMIT_PROPOSAL", "committee_verdict": "ADMIT_PROPOSAL", "judge_score": 0.90, "committee_outcome_binary": 1},
            {"proposal_id": "P2", "judge_verdict": "REJECT", "committee_verdict": "REJECT", "judge_score": 0.10, "committee_outcome_binary": 0},
            {"proposal_id": "P3", "judge_verdict": "ADMIT_PROPOSAL", "committee_verdict": "ADMIT_PROPOSAL", "judge_score": 0.85, "committee_outcome_binary": 1},
            {"proposal_id": "P4", "judge_verdict": "REJECT", "committee_verdict": "REJECT", "judge_score": 0.20, "committee_outcome_binary": 0}
        ]
    }

    result = run_judge_arbiter(input_data)
    calib = result["calibration_metrics"]
    assert calib["cohens_kappa"] == 1.0
    assert calib["brier_score"] <= 0.05
    assert "calibrated_rubric_weights" in calib
