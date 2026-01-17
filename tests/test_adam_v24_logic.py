import unittest
from unittest.mock import MagicMock, patch
import asyncio
from core.agents.market_sentiment_agent import MarketSentimentAgent
from core.prompting.personas.adam_risk_architect import AdamRiskArchitect

class TestAdamV24Logic(unittest.TestCase):
    def setUp(self):
        self.config = {'data_sources': []}

    @patch('core.agents.market_sentiment_agent.DataFetcher')
    def test_liquidity_mirage_trigger(self, MockDataFetcher):
        # Setup Mock
        mock_fetcher = MockDataFetcher.return_value

        # SPY Green (+1.0%)
        # QQQ Flat
        def fetch_market_side_effect(ticker):
            if ticker == "SPY":
                return {"current_price": 101.0, "previous_close": 100.0}
            if ticker == "QQQ":
                 return {"current_price": 100.0, "previous_close": 100.0}
            return {}

        mock_fetcher.fetch_market_data.side_effect = fetch_market_side_effect

        # HYG Red (-0.5%), IEF Flat (0.0%) -> Relative Spread -0.5% (Widening)
        mock_fetcher.fetch_credit_metrics.return_value = {
            "HYG": {"last_price": 99.5, "previous_close": 100.0}
        }
        mock_fetcher.fetch_treasury_metrics.return_value = {
            "IEF": {"last_price": 100.0, "previous_close": 100.0},
            "^TNX": {"last_price": 4.0},
            "^IRX": {"last_price": 3.0} # Normal curve
        }

        # Others empty
        mock_fetcher.fetch_volatility_metrics.return_value = {}
        mock_fetcher.fetch_crypto_metrics.return_value = {}
        mock_fetcher.fetch_macro_liquidity.return_value = {}

        # Run Agent
        agent = MarketSentimentAgent(self.config)
        agent.data_fetcher = mock_fetcher

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            trigger, details = loop.run_until_complete(agent.check_credit_dominance_rule())

            print("\n--- Test 1: Liquidity Mirage ---")
            print(f"Trigger: {trigger}")
            # print(f"Details: {details}")

            self.assertEqual(trigger, "Liquidity Mirage")
            self.assertGreater(details['spy_change_pct'], 0.5)
            self.assertLess(details['relative_credit_performance'], -0.15)

        finally:
            loop.close()

    @patch('core.agents.market_sentiment_agent.DataFetcher')
    def test_systemic_tremor_curve_inversion(self, MockDataFetcher):
        mock_fetcher = MockDataFetcher.return_value

        # Curve Inversion: 10Y (3.0%) < 3M (5.0%)
        mock_fetcher.fetch_treasury_metrics.return_value = {
            "^TNX": {"last_price": 3.0},
            "^IRX": {"last_price": 5.0},
            "IEF": {"last_price": 100.0, "previous_close": 100.0}
        }
        mock_fetcher.fetch_volatility_metrics.return_value = {
             "^VIX": {"last_price": 20.0},
             "^VIX3M": {"last_price": 22.0} # Normal VIX
        }
        mock_fetcher.fetch_credit_metrics.return_value = {"HYG": {"last_price": 100.0, "previous_close": 100.0}}
        mock_fetcher.fetch_crypto_metrics.return_value = {"BTC-USD": {"last_price": 100.0, "previous_close": 100.0}}
        mock_fetcher.fetch_market_data.return_value = {"current_price": 100.0, "previous_close": 100.0}
        mock_fetcher.fetch_macro_liquidity.return_value = {}

        agent = MarketSentimentAgent(self.config)
        agent.data_fetcher = mock_fetcher

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            trigger, details = loop.run_until_complete(agent.check_credit_dominance_rule())

            print("\n--- Test 2: Systemic Tremor (Curve Inversion) ---")
            print(f"Trigger: {trigger}")
            self.assertEqual(trigger, "Systemic Tremor")
            self.assertLess(details['curve_slope_10y_3m'], 0)
        finally:
            loop.close()

    @patch('core.agents.market_sentiment_agent.DataFetcher')
    def test_risk_on_decoupling_trigger(self, MockDataFetcher):
        mock_fetcher = MockDataFetcher.return_value

        def fetch_market_side_effect(ticker):
            if ticker == "QQQ":
                return {"current_price": 101.0, "previous_close": 100.0} # +1%
            if ticker == "SPY":
                return {"current_price": 100.0, "previous_close": 100.0}
            return {}
        mock_fetcher.fetch_market_data.side_effect = fetch_market_side_effect

        mock_fetcher.fetch_crypto_metrics.return_value = {
            "BTC-USD": {"last_price": 95.0, "previous_close": 100.0} # -5%
        }

        # Normal other metrics
        mock_fetcher.fetch_credit_metrics.return_value = {"HYG": {"last_price": 100.0, "previous_close": 100.0}}
        mock_fetcher.fetch_treasury_metrics.return_value = {"^TNX": {"last_price": 4.0}, "^IRX": {"last_price": 3.0}}
        mock_fetcher.fetch_volatility_metrics.return_value = {"^VIX": {"last_price": 20.0}, "^VIX3M": {"last_price": 22.0}}
        mock_fetcher.fetch_macro_liquidity.return_value = {}

        agent = MarketSentimentAgent(self.config)
        agent.data_fetcher = mock_fetcher

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            trigger, details = loop.run_until_complete(agent.check_credit_dominance_rule())

            print("\n--- Test 3: Risk-On Decoupling ---")
            print(f"Trigger: {trigger}")
            self.assertEqual(trigger, "Risk-On Decoupling")
        finally:
            loop.close()

    def test_persona_rendering(self):
        print("\n--- Test 4: Persona Prompt Rendering ---")
        persona = AdamRiskArchitect.default()
        inputs = {
            "market_data": "SPY +1.0%, HYG -0.5%, BTC -0.1%",
            "credit_status": "Liquidity Mirage"
        }
        rendered = persona.render(inputs)
        print("Rendered Prompt Content:")
        # print(rendered)
        self.assertIn("Liquidity Mirage", rendered)
        self.assertIn("SPY +1.0%", rendered)

if __name__ == '__main__':
    unittest.main()
