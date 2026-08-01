import pytest
import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from pydantic import ValidationError

from adam_os.core.events import LoanOriginated, DebtUpdated, AssetRevalued, CovenantEvaluated
from adam_os.contexts.ledger.aggregate import FinancialEntity
from adam_os.contexts.governance.engine import DeterministicPolicyEngine, FinancialContext
from adam_os.contexts.workflows.activities import evaluate_covenant, flag_asset
from adam_os.contexts.workflows.surveillance import PortfolioSurveillanceWorkflow

def test_aggregate_rebuilding_from_events() -> None:
    """Test Event Sourcing Aggregate Root rebuilding from event history."""
    entity_id = "ARM_LBO_01"
    
    # Using mixed aliases to prove the `populate_by_name` capability of the merged events
    events = [
        LoanOriginated(
            entity_id=entity_id, 
            principal_amount=300000.0, # HEAD branch alias mapping
            asset_value=1000000.0      # HEAD branch alias mapping
        ),
        DebtUpdated(
            entity_id=entity_id, 
            new_debt_amount=400000.0
        ),
        AssetRevalued(
            entity_id=entity_id, 
            new_asset_value_usd=800000.0 # main branch strict naming
        )
    ]

    entity = FinancialEntity.load_from_history(entity_id, events)

    assert entity.asset_value_usd == 800000.0
    assert entity.total_debt_usd == 400000.0
    assert entity.ltv == 0.5
    assert entity.version == 3
    assert len(entity.get_uncommitted_events()) == 0

def test_deterministic_policy_engine() -> None:
    """Test Deterministic Policy Engine catching breaches mathematically using both models and dicts."""
    ruleset = {
        "ltv_35": {
            "threshold": 0.35,
            "logic": {"<=": [{"var": "ltv"}, 0.35]}
        }
    }
    engine = DeterministicPolicyEngine(ruleset)

    # Test with strict Pydantic Model (20% LTV should pass)
    ctx_pass = FinancialContext(entity_id="ENT_1", asset_value_usd=1000000.0, total_debt_usd=200000.0)
    result_pass = engine.evaluate("ltv_35", ctx_pass)
    
    assert result_pass.passed is True
    assert result_pass.is_breached is False
    assert result_pass.alert_triggered is False
    assert result_pass.evaluated_value == 0.2

    # Test with flexible dictionary for dynamic expandability (40% LTV should fail)
    ctx_fail = {"entity_id": "ENT_2", "ltv": 0.40, "asset_value_usd": 1000000.0}
    result_fail = engine.evaluate("ltv_35", ctx_fail)
    
    assert result_fail.passed is False
    assert result_fail.is_breached is True
    assert result_fail.alert_triggered is True
    assert result_fail.evaluated_value == 0.40
    assert "evaluation_details" in result_fail.model_dump()

@pytest.mark.asyncio
async def test_continuous_surveillance_workflow() -> None:
    """Test the long-running Temporal PortfolioSurveillanceWorkflow logic."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="surveillance-queue",
            workflows=[PortfolioSurveillanceWorkflow],
            activities=[evaluate_covenant, flag_asset],
        ):
            handle = await env.client.start_workflow(
                PortfolioSurveillanceWorkflow.run,
                "asset-continuous", # entity_id argument
                id=f"surveillance-{uuid4()}",
                task_queue="surveillance-queue",
            )

            # Send a passing signal (safe asset context)
            await handle.signal(
                PortfolioSurveillanceWorkflow.reevaluate_asset, 
                {"target_rule": "softbank_arm_margin_loop", "ltv": 0.20}
            )
            
            # Send a breaching signal (failed asset context)
            await handle.signal(
                PortfolioSurveillanceWorkflow.reevaluate_asset, 
                {"target_rule": "softbank_arm_margin_loop", "ltv": 0.40}
            )

            # Send termination signal to break the infinite loop and allow completion
            await handle.signal(PortfolioSurveillanceWorkflow.terminate_surveillance)
            
            # Await workflow completion
            await handle.result()

            # Query the workflow state to prove continuous history was tracked correctly
            breaches = await handle.query(PortfolioSurveillanceWorkflow.get_breach_history)
            assert len(breaches) == 1
            assert breaches[0] == "softbank_arm_margin_loop"