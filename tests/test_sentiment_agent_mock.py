import pytest
import asyncio
from unittest.mock import MagicMock, patch
from core.agents.market_sentiment_agent import MarketSentimentAgent
from core.schemas.agent_schema import AgentInput, AgentOutput

@pytest.fixture
def agent():
    config = {
        "data_sources": ['news', 'prediction_market'],
        "sentiment_threshold": 0.5
    }
    with patch("core.agents.market_sentiment_agent.SimulatedFinancialNewsAPI", autospec=True), \
         patch("core.agents.market_sentiment_agent.SimulatedPredictionMarketAPI", autospec=True), \
         patch("core.agents.market_sentiment_agent.SimulatedSocialMediaAPI", autospec=True), \
         patch("core.agents.market_sentiment_agent.SimulatedWebTrafficAPI", autospec=True), \
         patch("core.agents.market_sentiment_agent.DataFetcher", autospec=True):
        agent = MarketSentimentAgent(config)

        # Mock internal methods to avoid complex logic if needed, or let them run if mocked APIs return good values
        agent.news_api.get_headlines.return_value = (["Headline"], 0.8)
        agent.prediction_market_api.get_market_sentiment.return_value = 0.7
        agent.social_media_api.get_social_media_sentiment.return_value = 0.6
        agent.web_traffic_api.get_web_traffic_sentiment.return_value = 0.5

        # Mock DataFetcher for check_credit_dominance_rule
        # This requires async methods
        # Ideally we mock check_credit_dominance_rule directly to simplify
        return agent

@pytest.mark.asyncio
async def test_sentiment_execution(agent):
    with patch.object(agent, 'check_credit_dominance_rule', return_value=(None, {})):
        input_data = AgentInput(query="Market", context={})
        result = await agent.execute(input_data)

        assert isinstance(result, AgentOutput)
        assert result.confidence > 0
        assert "Overall Sentiment Score" in result.answer
        assert result.metadata["sentiment_score"] > 0.0

@pytest.mark.asyncio
async def test_sentiment_override(agent):
    with patch.object(agent, 'check_credit_dominance_rule', return_value=("Systemic Tremor", {"vix": 30})):
        input_data = AgentInput(query="Market", context={})
        result = await agent.execute(input_data)

        assert isinstance(result, AgentOutput)
        # Should be overridden to 0.2
        assert result.metadata["sentiment_score"] == 0.2
        assert "Systemic Tremor" in result.answer
