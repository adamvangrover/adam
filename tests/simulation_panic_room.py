import pytest
import asyncio
from unittest.mock import MagicMock, patch
from core.agents.market_sentiment_agent import MarketSentimentAgent
from core.schemas.agent_schema import AgentInput, AgentOutput

# Simulation Parameters
CRISIS_VIX_LEVEL = 80.0
NORMAL_VIX_LEVEL = 15.0
INVERTED_YIELD_10Y = 1.5
INVERTED_YIELD_3M = 5.0

@pytest.fixture
def panic_agent():
    """
    Initializes a MarketSentimentAgent with mocked external APIs
    but REAL internal logic for credit dominance rules.
    """
    config = {
        "agent_id": "sentiment_agent_panic_test",
        "data_sources": ["news", "prediction_market"],
        "sentiment_threshold": 0.5
    }

    # Patch the external APIs
    with patch("core.agents.market_sentiment_agent.SimulatedFinancialNewsAPI") as MockNews, \
         patch("core.agents.market_sentiment_agent.SimulatedPredictionMarketAPI") as MockPred, \
         patch("core.agents.market_sentiment_agent.SimulatedSocialMediaAPI") as MockSocial, \
         patch("core.agents.market_sentiment_agent.SimulatedWebTrafficAPI") as MockWeb, \
         patch("core.agents.market_sentiment_agent.DataFetcher") as MockFetcher:

        agent = MarketSentimentAgent(config)

        # 1. Setup "Soft Data" to be deceptively BULLISH (to test override logic)
        # News says everything is great
        agent.news_api.get_headlines = MagicMock(return_value=(["Market Booms!", "All Time Highs"], 0.9))
        # Prediction markets are euphoric
        agent.prediction_market_api.get_market_sentiment = MagicMock(return_value=0.8)
        # Social media is hype
        agent.social_media_api.get_social_media_sentiment = MagicMock(return_value=0.9)
        # Web traffic is high
        agent.web_traffic_api.get_web_traffic_sentiment = MagicMock(return_value=0.8)

        # 2. Setup DataFetcher Mock
        # We will use side_effect in the tests to vary the market data
        agent.data_fetcher = MockFetcher.return_value

        yield agent

@pytest.mark.asyncio
async def test_systemic_tremor_vix_inversion(panic_agent):
    """
    Scenario: "Volmageddon 2.0"
    Soft data is bullish, but VIX (Spot) > VIX3M (3-Month).
    Expectation: Sentiment overrides to 0.2 (Bearish/Panic).
    """
    print("\n--- Simulation: Volmageddon 2.0 (VIX Inversion) ---")

    # Define the "Hard Data" state
    def mock_market_data(ticker):
        # Default benign data
        return {"symbol": ticker, "last_price": 100, "previous_close": 100}

    def mock_volatility_metrics():
        return {
            "^VIX": {"symbol": "^VIX", "last_price": CRISIS_VIX_LEVEL},     # 80.0
            "^VIX3M": {"symbol": "^VIX3M", "last_price": 40.0},             # 40.0 (Inverted!)
            "^MOVE": {"symbol": "^MOVE", "last_price": 150.0}
        }

    # Helper to return benign data for other calls
    panic_agent.data_fetcher.fetch_market_data.side_effect = mock_market_data
    panic_agent.data_fetcher.fetch_volatility_metrics.side_effect = mock_volatility_metrics
    panic_agent.data_fetcher.fetch_credit_metrics.return_value = {} # Benign
    panic_agent.data_fetcher.fetch_crypto_metrics.return_value = {}
    panic_agent.data_fetcher.fetch_treasury_metrics.return_value = {}
    panic_agent.data_fetcher.fetch_macro_liquidity.return_value = {}

    # Execute
    result = await panic_agent.execute(AgentInput(query="Assess Market Risk"))

    # Verification
    print(f"Agent Answer: {result.answer}")
    assert result.metadata["sentiment_score"] == 0.2
    assert result.metadata["details"]["status_override"] == "Systemic Tremor"
    assert "WARNING: Credit Dominance Rule Triggered: Systemic Tremor" in result.answer

@pytest.mark.asyncio
async def test_systemic_tremor_yield_curve_inversion(panic_agent):
    """
    Scenario: "Bond Market Flash Crash"
    Soft data is bullish, but 10Y Yield < 3M Yield (Deep Inversion).
    Expectation: Sentiment overrides to 0.2.
    """
    print("\n--- Simulation: Bond Market Flash Crash (Yield Inversion) ---")

    def mock_treasury_metrics():
        return {
            "^TNX": {"last_price": INVERTED_YIELD_10Y},  # 1.5%
            "^IRX": {"last_price": INVERTED_YIELD_3M},   # 5.0%
            "IEF": {"last_price": 100.0}
        }

    # Reset side effects
    panic_agent.data_fetcher.fetch_market_data.return_value = {"last_price": 100, "previous_close": 100}
    panic_agent.data_fetcher.fetch_volatility_metrics.return_value = {
        "^VIX": {"last_price": 15.0},
        "^VIX3M": {"last_price": 18.0} # Normal Contango
    }
    panic_agent.data_fetcher.fetch_treasury_metrics.side_effect = mock_treasury_metrics

    # Ensure other metrics are clean
    panic_agent.data_fetcher.fetch_credit_metrics.return_value = {}
    panic_agent.data_fetcher.fetch_crypto_metrics.return_value = {}
    panic_agent.data_fetcher.fetch_macro_liquidity.return_value = {}

    # Execute
    result = await panic_agent.execute(AgentInput(query="Assess Market Risk"))

    # Verification
    assert result.metadata["sentiment_score"] == 0.2
    assert result.metadata["details"]["status_override"] == "Systemic Tremor"

@pytest.mark.asyncio
async def test_liquidity_mirage(panic_agent):
    """
    Scenario: "Liquidity Mirage"
    Stocks are UP (Green), but Credit Spreads are BLOWING OUT.
    This is a classic trap (e.g., pre-2008 crash).
    Condition: SPY > +0.5% AND (HYG_Change - IEF_Change) < -0.15%
    """
    print("\n--- Simulation: Liquidity Mirage (Stocks Up, Credit Down) ---")

    # SPY is UP +1.0%
    spy_data = {"symbol": "SPY", "last_price": 505.0, "previous_close": 500.0} # +1.0%
    qqq_data = {"symbol": "QQQ", "last_price": 400.0, "previous_close": 400.0} # Flat

    def mock_market_data(ticker):
        if ticker == "SPY": return spy_data
        if ticker == "QQQ": return qqq_data
        return {}

    # HYG (Junk Bonds) crashing relative to IEF (Treasuries)
    # HYG: -2.0%
    # IEF: -0.5%
    # Spread Change: -2.0 - (-0.5) = -1.5% (Much less than -0.15% threshold)
    def mock_credit_metrics():
        return {
            "HYG": {"last_price": 98.0, "previous_close": 100.0}, # -2.0%
        }

    def mock_treasury_metrics():
        return {
            "IEF": {"last_price": 99.5, "previous_close": 100.0}, # -0.5%
            "^TNX": {"last_price": 4.0},
            "^IRX": {"last_price": 3.0} # Normal Curve
        }

    panic_agent.data_fetcher.fetch_market_data.side_effect = mock_market_data
    panic_agent.data_fetcher.fetch_credit_metrics.side_effect = mock_credit_metrics
    panic_agent.data_fetcher.fetch_treasury_metrics.side_effect = mock_treasury_metrics
    panic_agent.data_fetcher.fetch_volatility_metrics.return_value = {} # Normal

    # Ensure other metrics are clean
    panic_agent.data_fetcher.fetch_crypto_metrics.return_value = {}
    panic_agent.data_fetcher.fetch_macro_liquidity.return_value = {}

    # Execute
    result = await panic_agent.execute(AgentInput(query="Is the rally real?"))

    # Verification
    assert result.metadata["sentiment_score"] == 0.2
    assert result.metadata["details"]["status_override"] == "Liquidity Mirage"
    assert "Liquidity Mirage" in result.answer
