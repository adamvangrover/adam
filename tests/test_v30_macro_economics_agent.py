import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from core.v30_architecture.python_intelligence.agents.macro_economics_agent import MacroEconomicsAgent

@pytest.mark.asyncio
async def test_macro_economics_agent_emission():
    agent = MacroEconomicsAgent()
    agent.emit = AsyncMock()

    # Call the logic directly
    await agent.analyze_macro_state()

    assert agent.emit.call_count == 1
    call_args = agent.emit.call_args[0]
    assert call_args[0] == "macro_update"

    payload = call_args[1]
    assert "fed_funds_rate" in payload
    assert "cpi_yoy" in payload
    assert "gdp_growth" in payload
    assert "regime" in payload
    assert payload["regime"] in ["NEUTRAL", "RESTRICTIVE", "EXPANSIONARY", "STAGFLATION_RISK"]
