from datetime import datetime
from scripts.evaluate_estimator import EstimatorComplianceEvaluator

def test_evaluator_compliant():
    evaluator = EstimatorComplianceEvaluator()
    t_e = datetime(2026, 1, 1, 8, 0)
    t_k = datetime(2026, 1, 1, 10, 0)
    t_d = datetime(2026, 1, 1, 12, 0)
    t_x = datetime(2026, 1, 1, 14, 0)

    result = evaluator.evaluate_proposal("NONE", t_e, t_k, t_d, t_x, 0.1)
    assert result["compliant"] is True
    assert len(result["rejection_reasons"]) == 0
    assert result["epistemic_state"] == "SUPPORTED"

def test_evaluator_authority_violation():
    evaluator = EstimatorComplianceEvaluator()
    t_e = datetime(2026, 1, 1, 8, 0)
    t_k = datetime(2026, 1, 1, 10, 0)
    t_d = datetime(2026, 1, 1, 12, 0)
    t_x = datetime(2026, 1, 1, 14, 0)

    result = evaluator.evaluate_proposal("COMMIT", t_e, t_k, t_d, t_x, 0.1)
    assert result["compliant"] is False
    assert any("Authority Violation" in r for r in result["rejection_reasons"])

def test_evaluator_temporal_violation():
    evaluator = EstimatorComplianceEvaluator()
    t_e = datetime(2026, 1, 1, 12, 0) # t_e > t_k
    t_k = datetime(2026, 1, 1, 10, 0)
    t_d = datetime(2026, 1, 1, 14, 0)
    t_x = datetime(2026, 1, 1, 16, 0)

    result = evaluator.evaluate_proposal("NONE", t_e, t_k, t_d, t_x, 0.1)
    assert result["compliant"] is False
    assert any("Temporal Geometry Violation" in r for r in result["rejection_reasons"])

def test_evaluator_target_prompt():
    evaluator = EstimatorComplianceEvaluator()
    t_k = datetime(2026, 1, 1, 10, 0)
    t_e = datetime(2026, 1, 1, 12, 0) # Violation: t_e > t_k
    t_d = datetime(2026, 1, 1, 14, 0)
    t_x = datetime(2026, 1, 1, 16, 0)

    result = evaluator.evaluate_proposal("RECOMMEND_CAPITAL_ALLOCATION", t_e, t_k, t_d, t_x, 0.1)
    assert result["compliant"] is False
    assert len(result["rejection_reasons"]) >= 2
