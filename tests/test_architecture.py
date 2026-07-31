import pytest
import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from adam_os.core.events import LoanOriginated, DebtUpdated, AssetRevalued, CovenantEvaluated
from adam_os.contexts.ledger.aggregate import FinancialEntity
from adam_os.contexts.governance.engine import DeterministicPolicyEngine
from adam_os.contexts.workflows.activities import evaluate_covenant, flag_asset
from adam_os.contexts.workflows.surveillance import PortfolioSurveillanceWorkflow

def test_aggregate_rebuilding() -> None:
    """Test that the FinancialEntity aggregate correctly rebuilds state from events."""
    entity_id = "loan-123"
    entity = FinancialEntity(entity_id=entity_id)

    # Simulate a history of events
    events = [
        LoanOriginated(entity_id=entity_id, principal_amount=1000.0, asset_value=2000.0),
        DebtUpdated(entity_id=entity_id, new_debt_amount=1200.0), # LTV = 1200 / 2000 = 60%
        AssetRevalued(entity_id=entity_id, new_asset_value=3000.0) # LTV = 1200 / 3000 = 40%
    ]

    entity.load_from_history(events)

    assert entity.debt == 1200.0
    assert entity.asset_value == 3000.0
    assert entity.get_ltv() == 0.4
    assert entity.version == 3
    assert len(entity.get_uncommitted_events()) == 0

def test_policy_engine() -> None:
    """Test the deterministic policy engine evaluation logic using jsonLogic."""
    engine = DeterministicPolicyEngine()

    # Test safe LTV (30%)
    safe_context = {"debt": 300, "asset_value": 1000}
    safe_result = engine.evaluate("ltv_35", safe_context)
    assert safe_result.is_breached is False

    # Test breached LTV (40%)
    breach_context = {"debt": 400, "asset_value": 1000}
    breach_result = engine.evaluate("ltv_35", breach_context)
    assert breach_result.is_breached is True

@pytest.mark.asyncio
async def test_surveillance_workflow() -> None:
    """Test the Temporal PortfolioSurveillanceWorkflow logic."""
    async with await WorkflowEnvironment.start_time_skipping() as env:

        async with Worker(
            env.client,
            task_queue="surveillance-queue",
            workflows=[PortfolioSurveillanceWorkflow],
            activities=[evaluate_covenant, flag_asset],
        ):
            # Test a safe asset
            safe_handle = await env.client.start_workflow(
                PortfolioSurveillanceWorkflow.run,
                "asset-safe",
                id=f"surveillance-safe-{uuid4()}",
                task_queue="surveillance-queue",
            )

            await safe_handle.signal(PortfolioSurveillanceWorkflow.reevaluate_asset, arg={"debt": 300.0, "asset_value": 1000.0})
            await safe_handle.result()

            # Test a breached asset
            breached_handle = await env.client.start_workflow(
                PortfolioSurveillanceWorkflow.run,
                "asset-breached",
                id=f"surveillance-breached-{uuid4()}",
                task_queue="surveillance-queue",
            )

            await breached_handle.signal(PortfolioSurveillanceWorkflow.reevaluate_asset, arg={"debt": 400.0, "asset_value": 1000.0})
            await breached_handle.result()
            assert True
