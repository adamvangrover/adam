import pytest
from core.governance.sentinel_harness import CreditMetrics, GuardrailWrapper, run_credit_workflow

def test_credit_metrics_expected_loss():
    metrics = CreditMetrics(pd=0.05, lgd=0.5, ead=1000.0)
    assert metrics.expected_loss == 25.0

def test_guardrail_wrapper_pii():
    wrapper = GuardrailWrapper()
    sanitized = wrapper.sanitize_pii("Contact John Doe at 123-45-6789.")
    assert "John Doe" not in sanitized
    assert "[ENTITY_A]" in sanitized
    assert "123-45-6789" not in sanitized
    assert "[REDACTED_SSN]" in sanitized

def test_guardrail_wrapper_redline():
    wrapper = GuardrailWrapper()
    assert wrapper.enforce_redlines("Cayman Islands") is True
    assert wrapper.enforce_redlines("USA") is False

def test_run_credit_workflow_automated():
    # EL = 0.01 * 0.5 * 1000 = 5.0
    # Gate = 1000 * 0.15 = 150.0
    # Conviction = 0.95
    # el < gate and conviction > 0.9 -> AUTOMATED
    state, safe_prompt = run_credit_workflow(
        metrics_data={"pd": 0.01, "lgd": 0.5, "ead": 1000.0},
        conviction=0.95,
        npv_fees=1000.0,
        sigma=0.15,
        jurisdiction="USA",
        prompt="Test automated",
        context={}
    )
    assert state.routing_path == "AUTOMATED"
    assert state.requires_step_up is False

def test_run_credit_workflow_hitl():
    # EL = 0.5 * 0.5 * 1000 = 250.0
    # Gate = 1000 * 0.15 = 150.0
    # el >= gate -> HITL_TIER_3
    state, safe_prompt = run_credit_workflow(
        metrics_data={"pd": 0.5, "lgd": 0.5, "ead": 1000.0},
        conviction=0.95,
        npv_fees=1000.0,
        sigma=0.15,
        jurisdiction="USA",
        prompt="Test step up",
        context={}
    )
    assert state.routing_path == "HITL_TIER_3"
    assert state.requires_step_up is True

def test_run_credit_workflow_redline_breach():
    # EL is low, but jurisdiction is Cayman Islands
    state, safe_prompt = run_credit_workflow(
        metrics_data={"pd": 0.01, "lgd": 0.5, "ead": 100.0},
        conviction=0.99,
        npv_fees=1000.0,
        sigma=0.15,
        jurisdiction="Cayman Islands",
        prompt="Test redline",
        context={}
    )
    assert state.routing_path == "HITL_TIER_3"
    assert state.requires_step_up is True
    assert "[REDLINE BREACH: HIGH RISK JURISDICTION]" in safe_prompt
