import pytest
from core.schemas.sentinel import CreditMetrics
from core.governance.sentinel.harness import RiskSynthesisEngine, GuardrailWrapper

def test_risk_synthesis_engine_gate():
    engine = RiskSynthesisEngine()
    metrics = CreditMetrics(pd=0.1, lgd=0.5, ead=1000000)

    # EL = 0.1 * 0.5 * 1000000 = 50000
    # NPV Fees = 100000
    # Gate = 100000 * 0.15 = 15000
    # EL > Gate (50000 > 15000) -> requires step-up

    decision = engine.evaluate(metrics, npv_fees=100000, conviction_score=0.95)
    assert decision.requires_step_up is True
    assert decision.routing_path == "HITL_TIER_3"

def test_risk_synthesis_engine_pass():
    engine = RiskSynthesisEngine()
    metrics = CreditMetrics(pd=0.01, lgd=0.2, ead=100000)

    # EL = 0.01 * 0.2 * 100000 = 200
    # NPV Fees = 100000
    # Gate = 100000 * 0.15 = 15000
    # EL < Gate (200 < 15000) -> pass

    decision = engine.evaluate(metrics, npv_fees=100000, conviction_score=0.95)
    assert decision.requires_step_up is False
    assert decision.routing_path == "AUTOMATED"

def test_guardrail_sanitizer():
    guardrail = GuardrailWrapper()
    prompt = "Review John Doe with SSN 123-45-6789"
    sanitized = guardrail.sanitize_pii(prompt)
    assert "John Doe" not in sanitized
    assert "[ENTITY_A]" in sanitized
    assert "123-45-6789" not in sanitized
    assert "[SSN_REDACTED]" in sanitized

def test_guardrail_redline():
    guardrail = GuardrailWrapper()
    assert guardrail.enforce_redlines({"jurisdiction": "Cayman Islands"}) is True
    assert guardrail.enforce_redlines({"jurisdiction": "USA"}) is False
