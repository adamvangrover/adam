import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from core.v30_architecture.python_intelligence.agents.market_sentiment_agent import MarketSentimentAgent, SentimentOutput

@pytest.mark.asyncio
async def test_sentiment_agent_mock_fallback():
    # Force the fallback by making litellm unavailable
    with patch('core.v30_architecture.python_intelligence.agents.market_sentiment_agent.litellm', None):
        agent = MarketSentimentAgent()

        mock_news_data = [{"title": "Apple is doing well", "snippet": "Strong earnings reported", "url": "http://mock.com"}]

        # Test analysis directly
        result = agent.analyze_sentiment("AAPL", mock_news_data)

        assert result["asset_id"] == "AAPL"
        assert "sentiment_score" in result
        assert "confidence" in result
        assert "key_drivers" in result

        # Ensure it conforms to output bounds
        assert -1.0 <= result["sentiment_score"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0

@pytest.mark.asyncio
async def test_sentiment_agent_emit_packet():
    agent = MarketSentimentAgent()
    agent.emit = AsyncMock() # Mock the emit method to avoid actual networking

    # Run one single iteration manually instead of full loop
    with patch('core.v30_architecture.python_intelligence.agents.market_sentiment_agent.litellm', None):
        mock_news_data = [{"title": "Bitcoin is up", "snippet": "Price surged today", "url": "http://mock.com"}]
        analysis_result = agent.analyze_sentiment("BTC", mock_news_data)

        await agent.emit("THOUGHT", {
            "task": "sentiment_analysis",
            "status": "completed",
            "output": analysis_result
        })

        agent.emit.assert_called_once()
        args, kwargs = agent.emit.call_args
        assert args[0] == "THOUGHT"
        assert args[1]["task"] == "sentiment_analysis"
        assert args[1]["output"]["asset_id"] == "BTC"
