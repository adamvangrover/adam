import pytest
from pydantic import ValidationError
from adam_os.core.events import LoanOriginated, AssetRevalued
from adam_os.contexts.ledger.aggregate import FinancialEntity
from adam_os.contexts.governance.engine import DeterministicPolicyEngine, FinancialContext

def test_aggregate_rebuilding_from_events():
    """Test Event Sourcing Aggregate Root rebuilding from event history."""
    entity_id = "ARM_LBO_01"

    events = [
        LoanOriginated(
            entity_id=entity_id,
            asset_value_usd=1000000.0,
            total_debt_usd=300000.0
        ),
        AssetRevalued(
            entity_id=entity_id,
            new_asset_value_usd=800000.0
        )
    ]

    entity = FinancialEntity.load_from_history(entity_id, events)

    assert entity.asset_value_usd == 800000.0
    assert entity.total_debt_usd == 300000.0
    assert entity.ltv == 0.375

def test_deterministic_policy_engine_evaluates_jsonlogic():
    """Test Deterministic Policy Engine catching LTV breach mathematically."""
    ruleset = {
        "softbank_arm_margin_loop": {
            "threshold": 0.35,
            "logic": {"<=": [{"var": "ltv"}, 0.35]}
        }
    }

    engine = DeterministicPolicyEngine(ruleset)

    # 20% LTV should pass
    ctx_pass = FinancialContext(entity_id="ENT_1", asset_value_usd=1000000.0, total_debt_usd=200000.0)
    result_pass = engine.evaluate_covenant("softbank_arm_margin_loop", ctx_pass)
    assert result_pass.passed is True
    assert result_pass.alert_triggered is False

    # 40% LTV should fail and trigger alert
    ctx_fail = FinancialContext(entity_id="ENT_2", asset_value_usd=1000000.0, total_debt_usd=400000.0)
    result_fail = engine.evaluate_covenant("softbank_arm_margin_loop", ctx_fail)
    assert result_fail.passed is False
    assert result_fail.alert_triggered is True
    assert "Reflexive death spiral risk" in result_fail.details
    assert result_fail.evaluated_value == 0.4
