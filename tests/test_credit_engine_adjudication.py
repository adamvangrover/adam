"""
Pytest Verification Suite for Deterministic Credit Judge and Pydantic Interfaces.
Run: PYTHONPATH=$(pwd) uv run pytest tests/test_credit_engine_adjudication.py
"""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from scripts.deterministic_credit_judge import (
    DeterministicCreditJudge,
    ModelProposal,
    TemporalEnvelope,
)


def mock_envelope():
    return {
        "t_effective": datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc),
        "t_knowledge": datetime(2026, 10, 2, 8, 0, tzinfo=timezone.utc),
        "t_decision": datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc),
        "t_execution": datetime(2026, 10, 2, 9, 30, tzinfo=timezone.utc),
    }


def standard_proposals(ttc_bps=150, ltm_bps=185, fwd_bps=210, eq_bps=240):
    return [
        {"model_name": "TTC", "pd_1y_bps": ttc_bps, "confidence_permille": 850},
        {"model_name": "LTM", "pd_1y_bps": ltm_bps, "confidence_permille": 750},
        {"model_name": "FWD", "pd_1y_bps": fwd_bps, "confidence_permille": 700},
        {"model_name": "EQUITY", "pd_1y_bps": eq_bps, "confidence_permille": 900},
    ]


def test_nan_divergence_fails_closed():
    judge = DeterministicCreditJudge()
    bad_proposals = standard_proposals()
    bad_proposals[1]["pd_1y_bps"] = float("nan")

    result = judge.adjudicate(mock_envelope(), bad_proposals, "NORMAL", 1.0)
    assert result["circuit_breaker_status"] == "CIRCUIT_BREAKER_TRIPPED"
    assert result["error_reason"] == "NON_FINITE_NUMERIC_INPUT_DETECTED"


def test_exact_integer_bps_boundary_behavior():
    # 5.35% (535 bps) - 5.00% (500 bps) = 35 bps exact
    judge = DeterministicCreditJudge(tolerance_bps=35)
    proposals = standard_proposals(ttc_bps=500, ltm_bps=535, fwd_bps=550, eq_bps=600)

    result = judge.adjudicate(mock_envelope(), proposals, "NORMAL", 1.0)
    assert result["divergence_bps"] == 35
    assert result["capital_buffer_action"] == "APPLY_DOWNSIDE_SPREAD_SURCHARGE"
    assert result["underwriting_thesis_status"] == "DETERIORATING_FORWARD_RISK"


def test_normal_case_has_no_surcharge():
    judge = DeterministicCreditJudge(tolerance_bps=35)
    proposals = standard_proposals(ttc_bps=150, ltm_bps=160, fwd_bps=170, eq_bps=180)

    result = judge.adjudicate(mock_envelope(), proposals, "NORMAL", 1.0)
    assert result["divergence_bps"] == 10
    assert result["capital_buffer_action"] == "STANDARD_CAPITAL"
    assert result["underwriting_thesis_status"] == "FOUNDATIONAL_THESIS_INTACT"


def test_evaluation_ids_are_unique_and_non_constant():
    judge = DeterministicCreditJudge()
    proposals = standard_proposals()
    env = mock_envelope()

    ids = {judge.adjudicate(env, proposals, "NORMAL", 1.0)["evaluation_id"] for _ in range(5)}
    assert len(ids) == 5, "Evaluation IDs must be uniquely generated for every run"


def test_hash_is_not_empty_digest():
    judge = DeterministicCreditJudge()
    result = judge.adjudicate(mock_envelope(), standard_proposals(), "NORMAL", 1.0)
    empty_sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert result["decision_digest_sha256"] != empty_sha256
    assert len(result["decision_digest_sha256"]) == 64


def test_temporal_envelope_validation():
    valid_env = TemporalEnvelope(
        t_effective=datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc),
        t_knowledge=datetime(2026, 10, 2, 8, 0, tzinfo=timezone.utc),
        t_decision=datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc),
        t_execution=datetime(2026, 10, 2, 9, 30, tzinfo=timezone.utc),
    )
    assert valid_env.t_effective < valid_env.t_knowledge

    # Monotonicity failure
    with pytest.raises(ValidationError):
        TemporalEnvelope(
            t_effective=datetime(2026, 10, 5, 0, 0, tzinfo=timezone.utc),
            t_knowledge=datetime(2026, 10, 2, 8, 0, tzinfo=timezone.utc),
            t_decision=datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc),
            t_execution=datetime(2026, 10, 2, 9, 30, tzinfo=timezone.utc),
        )


def test_model_proposal_empty_sha256_rejection():
    empty_sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    valid_sha256 = "a" * 64

    ModelProposal(
        model_name="TTC",
        pd_1y_bps=150,
        confidence_permille=850,
        evidence_payload_sha256=valid_sha256,
        epistemic_state="SUPPORTED",
    )

    with pytest.raises(ValidationError):
        ModelProposal(
            model_name="TTC",
            pd_1y_bps=150,
            confidence_permille=850,
            evidence_payload_sha256=empty_sha256,
            epistemic_state="SUPPORTED",
        )
