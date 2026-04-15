import pytest
import asyncio
from core.v30_architecture.python_intelligence.agents.dynamic_search_agent import DynamicSearchAgent

@pytest.mark.asyncio
async def test_dynamic_search_agent_sweep_success():
    agent = DynamicSearchAgent()
    result = await agent.execute(
        target_asset="TEST_CORP",
        mode="SWEEP",
        edgar_success=True,
        dockets_success=True,
        current_hy_spread="400 bps"
    )

    assert "TEST_CORP" in result["result"]
    assert "2023-10-15" in result["result"]
    assert "400 bps" in result["result"]

@pytest.mark.asyncio
async def test_dynamic_search_agent_sweep_fallback():
    agent = DynamicSearchAgent()
    result = await agent.execute(
        target_asset="FAIL_CORP",
        mode="SWEEP",
        edgar_success=False,
        dockets_success=False,
        current_hy_spread="500 bps"
    )

    assert "FAIL_CORP" in result["result"]
    assert "NULL" in result["result"]
    assert "500 bps" in result["result"]

@pytest.mark.asyncio
async def test_dynamic_search_agent_deep_dive_success():
    agent = DynamicSearchAgent()
    result = await agent.execute(
        target_asset="TEST_CORP",
        mode="DEEP DIVE",
        edgar_success=True,
        dockets_success=True
    )

    report = result["result"]
    assert "# Distress Report: TEST_CORP" in report
    assert "Successfully searched site:sec.gov" in report
    assert "Successfully searched site:restructuring.ra.kroll.com" in report
    assert "Verification Checklist" in report
    assert len(result["fallbacks_used"]) == 0

@pytest.mark.asyncio
async def test_dynamic_search_agent_deep_dive_fallback():
    agent = DynamicSearchAgent()
    result = await agent.execute(
        target_asset="FAIL_CORP",
        mode="DEEP DIVE",
        edgar_success=False,
        dockets_success=False
    )

    report = result["result"]
    assert "# Distress Report: FAIL_CORP" in report
    assert "EDGAR delayed. Searched for trailing market proxies" in report
    assert "Dockets blocked/empty. Searched open-web financial press" in report
    assert "Verification Checklist" in report
    assert len(result["fallbacks_used"]) == 2
