"""
Unit and Property Tests for ADAM OS Policy Engine.
"""

import pytest
import json
from src.governance.policy_engine import (
    DeterministicPolicyEngine,
    FinancialContext,
    OPAAuthorizationClient
)

@pytest.fixture
def ruleset():
    return {
        "softbank_arm_margin_loop": {
            "threshold": 0.35,
            "logic": {"<=": [{"var": "ltv"}, 0.35]}
        }
    }

@pytest.fixture
def engine(ruleset):
    return DeterministicPolicyEngine(ruleset)

def test_financial_context_ltv_calculation():
    """Test deterministic math execution."""
    ctx = FinancialContext(entity_id="ARM_LBO_01", asset_value_usd=1000000.0, total_debt_usd=350000.0)
    assert ctx.ltv == 0.35

def test_policy_engine_passes_covenant(engine):
    """Test standard operational compliance."""
    ctx = FinancialContext(entity_id="ARM_LBO_01", asset_value_usd=1000000.0, total_debt_usd=200000.0) # 20% LTV
    result = engine.evaluate_covenant("softbank_arm_margin_loop", ctx)

    assert result.passed is True
    assert result.alert_triggered is False
    assert result.evaluated_value == 0.20

def test_policy_engine_fails_covenant_triggers_spiral(engine):
    """Test breach conditions and alert generation ('Reflexive death spiral')."""
    ctx = FinancialContext(entity_id="ARM_LBO_02", asset_value_usd=1000000.0, total_debt_usd=400000.0) # 40% LTV (Breach)
    result = engine.evaluate_covenant("softbank_arm_margin_loop", ctx)

    assert result.passed is False
    assert result.alert_triggered is True
    assert "Reflexive death spiral risk" in result.details

def test_opa_authorization_mock():
    """Ensure Sovereign Analyst role is properly gated."""
    assert OPAAuthorizationClient.is_authorized("sovereign_analyst", "execute_lbo") is True
    assert OPAAuthorizationClient.is_authorized("junior_analyst", "execute_lbo") is False
