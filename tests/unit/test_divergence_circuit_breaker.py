import pytest
from scripts.divergence_circuit_breaker import validate_model_divergence

def test_validate_model_divergence_pass():
    result = validate_model_divergence(pd_alpha=0.01, pd_beta=0.02, theta=0.05)
    assert result["circuit_breaker_tripped"] is False
    assert result["action"] == "PROCEED_TO_STATE_COMMIT"
    assert result["divergence_metric"] == 0.01

def test_validate_model_divergence_fail():
    result = validate_model_divergence(pd_alpha=0.01, pd_beta=0.07, theta=0.05)
    assert result["circuit_breaker_tripped"] is True
    assert result["action"] == "HALT_DIVERSE_TO_HITL"
    assert result["divergence_metric"] == 0.06

def test_validate_model_divergence_edge_case():
    result = validate_model_divergence(pd_alpha=0.01, pd_beta=0.06, theta=0.05)
    assert result["circuit_breaker_tripped"] is False
    assert result["action"] == "PROCEED_TO_STATE_COMMIT"
    assert result["divergence_metric"] == 0.05
