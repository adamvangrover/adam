import json
import os
from scripts.rubric_schema import MacroForecastSubmission, EvaluationRubricResult
from scripts.recursive_engine import EpistemicMemoryLedger, AppendOnlyAuditTrail, MacroAgentHarness

def mock_generator(prompt: str) -> str:
    # Simulated generation for the "Golden Exemplar"
    return json.dumps({
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
    })

def mock_judge(submission: MacroForecastSubmission) -> EvaluationRubricResult:
    # Simulated adversarial judge output
    return EvaluationRubricResult.model_validate({
        "challenge_id": submission.challenge_id,
        "causal_grounding": {
            "dimension": "Causal Grounding",
            "score": 5,
            "strengths": "Cites specific balance-sheet supply-absorption friction.",
            "deficiencies": "None noted.",
            "passed_gate": True
        },
        "transmission_mechanics": {
            "dimension": "Transmission Mechanics",
            "score": 5,
            "strengths": "Clear, mechanically traceable transmission path.",
            "deficiencies": "None noted.",
            "passed_gate": True
        },
        "counterfactual_falsification": {
            "dimension": "Counterfactual Falsification",
            "score": 4,
            "strengths": "Features two quantitative invalidation triggers.",
            "deficiencies": "Could specify exact time boundary.",
            "passed_gate": True
        },
        "epistemic_calibration": {
            "dimension": "Epistemic Calibration",
            "score": 5,
            "strengths": "Explicitly justifies why confidence is 0.74.",
            "deficiencies": "None noted.",
            "passed_gate": True
        }
    })

def mock_api_dispatch(payload: dict) -> dict:
    return {"status": "success", "receipt_id": "arena-12345"}

def run_simulation():
    memory = EpistemicMemoryLedger(memory_file_path="data/calibration_memory.json")
    audit = AppendOnlyAuditTrail(ledger_dir="forecast_ledger")
    harness = MacroAgentHarness(memory, audit, mock_generator, mock_judge, mock_api_dispatch)

    market_challenge = {
        "challenge_id": "HA-20260928-UST10Y-SETTLE",
        "target_asset": "US_10Y_BENCHMARK_YIELD",
        "market_state": {"spot_yield": 4.382}
    }

    result = harness.execute_pipeline(
        base_directive="SYSTEM DIRECTIVE: ADAM-MACRO-SENTINEL",
        market_challenge=market_challenge,
        timestamp_str="20260927T154612Z"
    )

    print("Pipeline Execution Complete:")
    print(json.dumps(result, indent=2))
    
    print("Recording settlement...")
    memory.record_settlement(
        challenge_id="HA-20260928-UST10Y-SETTLE",
        asset="US_10Y_BENCHMARK_YIELD",
        predicted_dir="bullish",
        confidence=0.74,
        actual_outcome="bullish",
        actual_settlement_price=4.438,
        point_forecast=4.445,
        judge_score=93.75,
        post_mortem_insight="Dealer balance sheet friction transmission functioned as modeled."
    )
    print("Settlement recorded.")

if __name__ == "__main__":
    run_simulation()
