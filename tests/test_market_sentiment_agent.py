from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agents.market_sentiment_agent import MarketSentimentAgent


@pytest.fixture
def agent_config():
    return {
        'data_sources': ['news', 'prediction_market', 'social_media', 'web_traffic'],
        'sentiment_threshold': 0.5
    }

@pytest.fixture
def sentiment_agent(agent_config):
    return MarketSentimentAgent(agent_config)

@pytest.mark.asyncio
async def test_analyze_sentiment_standard_execution(sentiment_agent):
    # Mock all the synchronous sources
    sentiment_agent.news_api.get_headlines = MagicMock(return_value=(["Headline"], 0.6))
    sentiment_agent.prediction_market_api.get_market_sentiment = MagicMock(return_value=0.7)
    sentiment_agent.social_media_api.get_social_media_sentiment = MagicMock(return_value=0.5)
    sentiment_agent.web_traffic_api.get_web_traffic_sentiment = MagicMock(return_value=0.4)

    # Mock the internal logic check_credit_dominance_rule to not trigger
    with patch.object(sentiment_agent, 'check_credit_dominance_rule', new_callable=AsyncMock) as mock_rule:
        mock_rule.return_value = (None, {})
        overall, details = await sentiment_agent.analyze_sentiment()

        assert overall > 0.0
        assert details["news_sentiment"] == 0.6
        assert details["prediction_market_sentiment"] == 0.7

@pytest.mark.asyncio
async def test_systemic_tremor_triggered_by_arbitrage(sentiment_agent):
    # Setup mocks for data fetcher
    sentiment_agent.data_fetcher.fetch_market_data = MagicMock(return_value={"SPY": {"last_price": 100, "previous_close": 100}, "QQQ": {"last_price": 100, "previous_close": 100}})
    sentiment_agent.data_fetcher.fetch_credit_metrics = MagicMock(return_value={"HYG": {"last_price": 100, "previous_close": 100}})
    sentiment_agent.data_fetcher.fetch_volatility_metrics = MagicMock(return_value={"^VIX": {"last_price": 15}, "^VIX3M": {"last_price": 20}}) # Normal VIX curve
    sentiment_agent.data_fetcher.fetch_crypto_metrics = MagicMock(return_value={"BTC-USD": {"last_price": 50000, "previous_close": 50000}})
    sentiment_agent.data_fetcher.fetch_treasury_metrics = MagicMock(return_value={"IEF": {"last_price": 100, "previous_close": 100}, "^TNX": {"last_price": 4.0}, "^IRX": {"last_price": 3.0}}) # Normal yield curve
    sentiment_agent.data_fetcher.fetch_macro_liquidity = MagicMock(return_value={"liquidity_index": 100})

    # Mock Arbitrage Agent to return opportunities
    mock_arbitrage_output = {
        "opportunities": [{"symbol": "BTC/USDT", "spread_percentage": 2.5}],
        "status": "success"
    }

    with patch.object(sentiment_agent.crypto_arbitrage_agent, 'execute', new_callable=AsyncMock) as mock_arbitrage:
        mock_arbitrage.return_value = mock_arbitrage_output

        trigger, details = await sentiment_agent.check_credit_dominance_rule()

        assert trigger == "Systemic Tremor"
        assert details.get("arbitrage_opportunities") == 1

@pytest.mark.asyncio
async def test_sentiment_score_overridden_by_systemic_tremor(sentiment_agent):
    sentiment_agent.news_api.get_headlines = MagicMock(return_value=(["Headline"], 0.9))
    sentiment_agent.prediction_market_api.get_market_sentiment = MagicMock(return_value=0.9)
    sentiment_agent.social_media_api.get_social_media_sentiment = MagicMock(return_value=0.9)
    sentiment_agent.web_traffic_api.get_web_traffic_sentiment = MagicMock(return_value=0.9)

    with patch.object(sentiment_agent, 'check_credit_dominance_rule', new_callable=AsyncMock) as mock_rule:
        mock_rule.return_value = ("Systemic Tremor", {"arbitrage_opportunities": 1})
        overall, details = await sentiment_agent.analyze_sentiment()

        assert overall == 0.2
        assert details.get("status_override") == "Systemic Tremor"
