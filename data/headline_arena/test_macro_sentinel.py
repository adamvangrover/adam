import pytest
import os
import tempfile
import json
from scripts.rubric_schema import RationaleComponentScore, EvaluationRubricResult, MacroForecastSubmission
from scripts.recursive_engine import EpistemicMemoryLedger

def test_macro_forecast_submission_valid():
    data = {
        "challenge_id": "HA-20260928-UST10Y-SETTLE",
        "target_asset": "US_10Y_BENCHMARK_YIELD",
        "direction": "bullish",
        "confidence": 0.74,
        "point_forecast": 4.445,
        "std_deviation": 0.038,
        "rationale": {
            "causal_grounding": "Primary catalyst is supply-demand indigestion driven by the Treasury's backloaded $183B coupon settlement slate coinciding with Q3 month-end commercial bank balance sheet optimization. Primary dealer net long duration positioning sits in the 88th percentile trailing 12-month window, creating a structural absorption constraint rather than speculative sentiment-driven flow.",
            "transmission_mechanism": "Auction concession concessions will force primary dealers to cheapen cash bonds on curve to clear balance sheet constraints prior to Tuesday closing. As dealer risk-weighted asset (RWA) limits bind under Supplementary Leverage Ratio (SLR) quarter-end reporting, the repo basis widens (+3.5 bps in SOFR-Treasury spread). This mechanics forces concession concessions through the 10-year belly, transmitting auction supply indigestion directly into a 5-7 bps term premium expansion.",
            "counterfactual_falsification": "Thesis is invalidated if the 2-year yield rallies >6 bps below 3.78% or if overnight reverse repo (ON RRP) facility drainage accelerates above $45B in a single window, neutralizing private-dealer balance sheet absorption friction. An immediate rotation out of high-beta tech into safe-haven benchmark duration before 13:00 EST would mechanically nullify the upward term premium pressure.",
            "calibration_and_sizing": "Confidence calibrated at 0.74 reflecting high conviction on structural dealer balance sheet mechanics, but capped below 0.80 due to binary quarter-end duration rebalancing by passive indexers (Barclays Aggregate extension trades). Implied volatility via MOVE index (98.4) supports a 1-day standard deviation boundary of +/- 3.8 bps against our point forecast of 4.445%."
        },
        "summary_statement": "Heavy Treasury coupon absorption into quarter-end dealer balance sheet constraints will force an upward term premium concession toward 4.445%. Conviction is sized at 0.74, bounded by passive month-end duration extension demand."
    }
    submission = MacroForecastSubmission.model_validate(data)
    assert submission.challenge_id == "HA-20260928-UST10Y-SETTLE"
    assert submission.confidence == 0.74

def test_evaluation_rubric_result_calculation():
    data = {
        "challenge_id": "HA-20260928-UST10Y-SETTLE",
        "causal_grounding": {"dimension": "Causal Grounding", "score": 5, "strengths": "Good", "deficiencies": "None", "passed_gate": True},
        "transmission_mechanics": {"dimension": "Transmission Mechanics", "score": 5, "strengths": "Good", "deficiencies": "None", "passed_gate": True},
        "counterfactual_falsification": {"dimension": "Counterfactual Falsification", "score": 4, "strengths": "Good", "deficiencies": "None", "passed_gate": True},
        "epistemic_calibration": {"dimension": "Epistemic Calibration", "score": 5, "strengths": "Good", "deficiencies": "None", "passed_gate": True}
    }
    rubric = EvaluationRubricResult.model_validate(data)
    assert rubric.aggregate_score == 93.75
    assert rubric.audit_verdict == "APPROVE"
    assert len(rubric.rejection_reasons) == 0

def test_evaluation_rubric_result_rejection():
    data = {
        "challenge_id": "HA-123",
        "causal_grounding": {"dimension": "Causal Grounding", "score": 2, "strengths": "Bad", "deficiencies": "Deficient", "passed_gate": False},
        "transmission_mechanics": {"dimension": "Transmission Mechanics", "score": 5, "strengths": "Good", "deficiencies": "None", "passed_gate": True},
        "counterfactual_falsification": {"dimension": "Counterfactual Falsification", "score": 4, "strengths": "Good", "deficiencies": "None", "passed_gate": True},
        "epistemic_calibration": {"dimension": "Epistemic Calibration", "score": 5, "strengths": "Good", "deficiencies": "None", "passed_gate": True}
    }
    rubric = EvaluationRubricResult.model_validate(data)
    assert rubric.aggregate_score == 75.0
    assert rubric.audit_verdict == "REJECT_FOR_REFINEMENT"
    assert len(rubric.rejection_reasons) > 0

def test_epistemic_memory_ledger():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = os.path.join(tmpdir, "memory.json")
        ledger = EpistemicMemoryLedger(memory_path)
        ledger.record_settlement(
            challenge_id="HA-TEST",
            asset="US_10Y",
            predicted_dir="bullish",
            confidence=0.74,
            actual_outcome="bullish",
            actual_settlement_price=4.438,
            point_forecast=4.445,
            judge_score=93.75,
            post_mortem_insight="Insight"
        )
        assert os.path.exists(memory_path)
        with open(memory_path, "r") as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["challenge_id"] == "HA-TEST"
        context = ledger.generate_calibration_context()
        assert "HISTORICAL BASE RATE: 100.0%" in context
