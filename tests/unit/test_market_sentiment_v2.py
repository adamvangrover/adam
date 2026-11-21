import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict
from core.agents.market_sentiment_agent import MarketSentimentAgent
from core.schemas.agent_schema import AgentInput, AgentOutput

@pytest.fixture
def sentiment_agent():
    config = {
        "agent_id": "sentiment_agent",
        "data_sources": ["news", "prediction_market"],
        "sentiment_threshold": 0.5
    }

    with patch("core.agents.market_sentiment_agent.SimulatedFinancialNewsAPI"), \
         patch("core.agents.market_sentiment_agent.SimulatedPredictionMarketAPI"), \
         patch("core.agents.market_sentiment_agent.SimulatedSocialMediaAPI"), \
         patch("core.agents.market_sentiment_agent.SimulatedWebTrafficAPI"), \
         patch("core.agents.market_sentiment_agent.DataFetcher"):

        agent = MarketSentimentAgent(config)

        # Mock API calls
        agent.news_api.get_headlines = MagicMock(return_value=([], 0.8))
        agent.prediction_market_api.get_market_sentiment = MagicMock(return_value=0.6)
        agent.social_media_api.get_social_media_sentiment = MagicMock(return_value=0.7)
        agent.web_traffic_api.get_web_traffic_sentiment = MagicMock(return_value=0.5)

        # Mock Credit Dominance Rule to avoid complex data fetching during unit test unless needed
        agent.check_credit_dominance_rule = AsyncMock(return_value=(None, {}))

        return agent

@pytest.mark.asyncio
async def test_legacy_execution(sentiment_agent):
    result = await sentiment_agent.execute()
    assert isinstance(result, dict)
    assert "sentiment_score" in result
    assert result["sentiment_score"] > 0.5

@pytest.mark.asyncio
async def test_standard_execution(sentiment_agent):
    input_obj = AgentInput(query="Analyze current market sentiment")
    result = await sentiment_agent.execute(input_obj)

    assert isinstance(result, AgentOutput)
    assert result.confidence > 0.0
    assert "Market Sentiment Analysis" in result.answer or "score" in result.answer
    assert result.metadata["sentiment_score"] > 0.0

@pytest.mark.asyncio
async def test_credit_dominance_trigger(sentiment_agent):
    # Force credit dominance trigger
    sentiment_agent.check_credit_dominance_rule = AsyncMock(return_value=("Systemic Tremor", {"vix": 30}))

    input_obj = AgentInput(query="Check risks")
    result = await sentiment_agent.execute(input_obj)

    assert isinstance(result, AgentOutput)
    assert result.metadata["details"]["status_override"] == "Systemic Tremor"
    # Expect sentiment to be lowered/overridden
    assert result.metadata["sentiment_score"] == 0.2
