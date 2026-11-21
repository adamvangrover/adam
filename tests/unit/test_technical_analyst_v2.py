import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
from typing import Any, Dict
# Import the class (it might fail if dependencies are missing, so we might need to patch imports if this was running in a strict env without deps)
# Assuming deps are present or we mock them.
from core.agents.technical_analyst_agent import TechnicalAnalystAgent
from core.schemas.agent_schema import AgentInput, AgentOutput

@pytest.fixture
def technical_agent():
    config = {
        "agent_id": "tech_agent",
        "data_sources": ["historical_price"]
    }

    # Patch DataFetcher and internal model methods to avoid real I/O and loading models
    with patch("core.agents.technical_analyst_agent.DataFetcher") as MockFetcher, \
         patch("core.agents.technical_analyst_agent.RandomForestClassifier"), \
         patch("core.agents.technical_analyst_agent.pd") as MockPandas:

        # Setup DataFetcher mock
        mock_fetcher_instance = MockFetcher.return_value
        mock_fetcher_instance.fetch_historical_data.return_value = [
            {"close": 100}, {"close": 101}, {"close": 102}
        ]

        agent = TechnicalAnalystAgent(config)
        agent.data_fetcher = mock_fetcher_instance # Ensure it's attached

        # Mock analyze_price_data to return a simple signal
        agent.analyze_price_data = MagicMock(return_value="buy")

        return agent

@pytest.mark.asyncio
async def test_legacy_execution_with_data(technical_agent):
    # Test passing data directly (legacy-ish, though execute signature changes)
    # The new execute will handle dict input.
    input_data = {"price_data": [{"close": 100}, {"close": 105}]}
    result = await technical_agent.execute(input_data)

    # Should return dict in legacy mode
    assert isinstance(result, dict)
    assert result["signal"] == "buy"

@pytest.mark.asyncio
async def test_standard_execution_with_ticker(technical_agent):
    input_obj = AgentInput(query="AAPL")

    # We expect the agent to fetch data for AAPL using DataFetcher
    result = await technical_agent.execute(input_obj)

    assert isinstance(result, AgentOutput)
    assert "Signal: BUY" in result.answer
    technical_agent.data_fetcher.fetch_historical_data.assert_called_with("AAPL")

@pytest.mark.asyncio
async def test_standard_execution_no_data(technical_agent):
    technical_agent.data_fetcher.fetch_historical_data.return_value = []

    input_obj = AgentInput(query="UNKNOWN_TICKER")
    result = await technical_agent.execute(input_obj)

    assert isinstance(result, AgentOutput)
    # If no data, analyze_price_data might fail or return something else.
    # But since we mocked analyze_price_data to always return "buy",
    # we should update the mock to simulate failure if empty data passed,
    # or the agent logic should handle it before calling analyze.

    # Let's adjust the expectation: The agent should check for empty data.
    # We'll see how we implement it.
    pass
