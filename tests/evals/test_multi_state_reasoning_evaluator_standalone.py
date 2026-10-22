import pytest
from evals.graders.multi_state_reasoning_evaluator import MultiStateReasoningEvaluator

def test_trace_fidelity_exact_match():
    trace = ["plan", "retrieve", "reason", "verify"]
    score = MultiStateReasoningEvaluator.evaluate_trace_fidelity(trace, trace)
    assert score == 1.0

def test_trace_fidelity_mismatch():
    pred = ["plan", "retrieve", "hallucinate"]
    gt = ["plan", "retrieve", "reason", "verify"]
    score = MultiStateReasoningEvaluator.evaluate_trace_fidelity(pred, gt)
    assert 0.0 <= score < 1.0

def test_fact_insertion():
    expected = [{"fact_id": "f1"}, {"fact_id": "f2"}, {"fact_id": "f3"}]
    inserted = [{"fact_id": "f1"}, {"fact_id": "f2"}, {"fact_id": "f4"}]
    metrics = MultiStateReasoningEvaluator.evaluate_fact_insertion(inserted, expected)

    assert metrics["precision"] == pytest.approx(2/3, abs=1e-3)
    assert metrics["recall"] == pytest.approx(2/3, abs=1e-3)
    assert metrics["f1"] == pytest.approx(2/3, abs=1e-3)

def test_reasoning_rubric():
    payload = {
        "evidence": ["E1", "E2", "E3", "E4", "E5"],
        "gaps_identified": ["G1", "G2", "G3"],
        "monotonic_confidence_steps": [0.5, 0.7, 0.85, 0.95]
    }
    rubric = MultiStateReasoningEvaluator.evaluate_reasoning_rubric(payload)
    assert rubric["evidence_grounding"] == 1.0
    assert rubric["gap_identification"] == 1.0
    assert rubric["monotonicity"] == 1.0
    assert rubric["composite_rubric_score"] == 1.0

def test_expected_calibration_error():
    confidences = [0.1, 0.2, 0.8, 0.9]
    accuracies = [0, 0, 1, 1]
    ece = MultiStateReasoningEvaluator.calculate_expected_calibration_error(confidences, accuracies, num_bins=2)
    assert ece == pytest.approx(0.15, abs=0.05)

def test_evaluate_multi_state_run():
    evaluator = MultiStateReasoningEvaluator()
    payload = {
        "predicted_trace": ["plan", "retrieve", "reason"],
        "ground_truth_trace": ["plan", "retrieve", "reason"],
        "inserted_facts": [{"fact_id": "f1"}],
        "expected_facts": [{"fact_id": "f1"}],
        "reasoning_payload": {
            "evidence": ["E1", "E2", "E3", "E4", "E5"],
            "gaps_identified": ["G1", "G2", "G3"],
            "monotonic_confidence_steps": [0.5, 0.8]
        },
        "confidences": [0.9, 0.95],
        "accuracies": [1, 1]
    }
    res = evaluator.evaluate_multi_state_run(payload)
    assert res["trace_fidelity"] == 1.0
    assert res["fact_insertion"]["f1"] == 1.0
    assert res["overall_quality_score"] > 0.9
