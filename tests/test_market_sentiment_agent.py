import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from core.agents.market_sentiment_agent import MarketSentimentAgent
from core.schemas.agent_schema import AgentOutput

@pytest.fixture
def mock_crypto_agent():
    with patch('core.agents.market_sentiment_agent.CryptoArbitrageAgent') as mock:
        yield mock

@pytest.fixture
def mock_fetcher():
    with patch('core.agents.market_sentiment_agent.DataFetcher') as mock:
        yield mock

@pytest.fixture
def mock_sources():
    with patch('core.agents.market_sentiment_agent.SimulatedFinancialNewsAPI') as news_mock, \
         patch('core.agents.market_sentiment_agent.SimulatedPredictionMarketAPI') as pred_mock, \
         patch('core.agents.market_sentiment_agent.SimulatedSocialMediaAPI') as social_mock, \
         patch('core.agents.market_sentiment_agent.SimulatedWebTrafficAPI') as web_mock:

         news_instance = news_mock.return_value
         news_instance.get_headlines.return_value = ([], 0.8) # Bullish news

         pred_instance = pred_mock.return_value
         pred_instance.get_market_sentiment.return_value = 0.7

         social_instance = social_mock.return_value
         social_instance.get_social_media_sentiment.return_value = 0.6

         web_instance = web_mock.return_value
         web_instance.get_web_traffic_sentiment.return_value = 0.5

         yield

@pytest.mark.asyncio
async def test_sentiment_no_arbitrage(mock_crypto_agent, mock_fetcher, mock_sources):
    # Setup crypto agent to return no opportunities
    crypto_instance = mock_crypto_agent.return_value
    # Fix: Provide required fields for AgentOutput
    crypto_instance.execute = AsyncMock(return_value=AgentOutput(
        answer="No opportunities",
        confidence=1.0,
        metadata={"opportunities": []}
    ))

    # Setup DataFetcher to return safe values (avoid credit trigger)
    fetcher_instance = mock_fetcher.return_value
    fetcher_instance.fetch_market_data.return_value = {"last_price": 100, "previous_close": 99.9} # Flat
    fetcher_instance.fetch_credit_metrics.return_value = {}
    fetcher_instance.fetch_volatility_metrics.return_value = {}
    fetcher_instance.fetch_crypto_metrics.return_value = {}
    fetcher_instance.fetch_treasury_metrics.return_value = {}
    fetcher_instance.fetch_macro_liquidity.return_value = {}

    config = {'data_sources': ['news']}
    agent = MarketSentimentAgent(config)

    # Use AgentInput explicitly to ensure standard output (AgentOutput object)
    # The current implementation returns AgentOutput only if input is AgentInput
    # or if we detect standard mode.
    # The previous test failed with "dict object has no attribute metadata" because
    # execute("Status") (string) might be returning a dict in legacy mode unless we force it.
    # In `execute`:
    # if isinstance(input_data, str): query = input_data; is_standard_mode = True (This was added in my fix to CryptoAgent, but MarketSentimentAgent handles it differently)

    # Looking at MarketSentimentAgent.execute:
    # if isinstance(input_data, str): query = input_data
    # It does NOT set is_standard_mode = True for string.
    # It only sets is_standard_mode = True if isinstance(input_data, AgentInput).

    # So we must pass AgentInput to get an AgentOutput object.

    from core.schemas.agent_schema import AgentInput
    result = await agent.execute(AgentInput(query="Status"))

    # Score should be high (0.8 news, 0.7 pred...) -> ~0.7
    assert result.metadata['sentiment_score'] > 0.6
    assert "arbitrage_override" not in result.metadata['details']

@pytest.mark.asyncio
async def test_sentiment_with_high_arbitrage(mock_crypto_agent, mock_fetcher, mock_sources):
    # Setup crypto agent to return MANY opportunities (triggering override)
    crypto_instance = mock_crypto_agent.return_value
    # Simulating 5 opportunities
    ops = [{"symbol": "BTC"} for _ in range(5)]
    # Fix: Provide required fields for AgentOutput
    crypto_instance.execute = AsyncMock(return_value=AgentOutput(
        answer="Found 5 opportunities",
        confidence=1.0,
        metadata={"opportunities": ops}
    ))

    # Setup DataFetcher safe values
    fetcher_instance = mock_fetcher.return_value
    fetcher_instance.fetch_market_data.return_value = {"last_price": 100, "previous_close": 99.9}
    fetcher_instance.fetch_credit_metrics.return_value = {}
    fetcher_instance.fetch_volatility_metrics.return_value = {}
    fetcher_instance.fetch_crypto_metrics.return_value = {}
    fetcher_instance.fetch_treasury_metrics.return_value = {}
    fetcher_instance.fetch_macro_liquidity.return_value = {}

    config = {'data_sources': ['news']}
    agent = MarketSentimentAgent(config)

    from core.schemas.agent_schema import AgentInput
    result = await agent.execute(AgentInput(query="Status"))

    # Score should be dampened by 0.2
    # Base calculation: (0.8*0.4 + 0.7*0.3 + 0.6*0.2 + 0.5*0.1) = 0.32 + 0.21 + 0.12 + 0.05 = 0.7
    # With dampen: 0.5

    expected_score = 0.7 - 0.2
    assert abs(result.metadata['sentiment_score'] - expected_score) < 0.01
    assert result.metadata['details']['arbitrage_override'] == "High Dislocation"
    assert "Crypto Inefficiency Detected" in result.answer
