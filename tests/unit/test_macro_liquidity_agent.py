import unittest
import asyncio
from unittest.mock import MagicMock, patch
from core.agents.specialized.macro_liquidity_agent import MacroLiquidityAgent

class TestMacroLiquidityAgent(unittest.TestCase):
    def setUp(self):
        self.config = {}
        self.agent = MacroLiquidityAgent(config=self.config)

    @patch('core.agents.specialized.macro_liquidity_agent.DataFetcher')
    def test_execute_normal_liquidity(self, mock_fetcher_cls):
        # Setup Mock
        mock_instance = mock_fetcher_cls.return_value
        self.agent.data_fetcher = mock_instance # Inject mock

        # Mock Data for Normal/Neutral Scenario
        # TNX=4.0, IRX=3.0 (Steep/Normal Slope > 0.5)
        # HYG=75, LQD=105 -> Ratio ~0.71 (Normal)
        # DXY=102 (Neutral)

        def mock_snapshot(symbol):
            data_map = {
                "^TNX": {"last_price": 4.0},
                "^IRX": {"last_price": 3.0},
                "HYG": {"last_price": 75.0},
                "LQD": {"last_price": 105.0},
                "DX-Y.NYB": {"last_price": 102.0},
                "GC=F": {"last_price": 2000.0}
            }
            return data_map.get(symbol, {})

        mock_instance.fetch_realtime_snapshot.side_effect = mock_snapshot
        # fetch_market_data shouldn't be called if snapshot succeeds, but mock it just in case
        mock_instance.fetch_market_data.return_value = {}

        # Run
        result = asyncio.run(self.agent.execute())

        # Verify
        self.assertEqual(result["regime"], "EXPANSIONARY")
        self.assertGreater(result["liquidity_score"], 20)
        self.assertLess(result["liquidity_score"], 50)
        self.assertEqual(result["confidence_score"], 1.0)

    @patch('core.agents.specialized.macro_liquidity_agent.DataFetcher')
    def test_execute_crisis_scenario(self, mock_fetcher_cls):
        # Setup Mock
        mock_instance = mock_fetcher_cls.return_value
        self.agent.data_fetcher = mock_instance

        # Mock Data for Crisis
        # TNX=5.0 (High Rate Stress)
        # Yield Curve Inverted: TNX=5.0, IRX=5.5
        # HYG=65, LQD=105 -> Ratio ~0.61 (Blowout)
        # DXY=110 (Strong Dollar Crush)

        def mock_snapshot(symbol):
            data_map = {
                "^TNX": {"last_price": 5.0},
                "^IRX": {"last_price": 5.5},
                "HYG": {"last_price": 65.0},
                "LQD": {"last_price": 105.0},
                "DX-Y.NYB": {"last_price": 110.0},
                "GC=F": {"last_price": 2200.0}
            }
            return data_map.get(symbol, {})

        mock_instance.fetch_realtime_snapshot.side_effect = mock_snapshot

        # Run
        result = asyncio.run(self.agent.execute())

        # Verify
        self.assertIn("CRISIS", result["regime"])
        self.assertGreater(result["liquidity_score"], 75)
        self.assertIn("Yield Curve Inverted", result["reasoning_trace"])
        self.assertIn("Credit Spreads Widening", result["reasoning_trace"])

    @patch('core.agents.specialized.macro_liquidity_agent.DataFetcher')
    def test_missing_data_confidence(self, mock_fetcher_cls):
        # Setup Mock to return empty for half tickers
        mock_instance = mock_fetcher_cls.return_value
        self.agent.data_fetcher = mock_instance

        def mock_snapshot(symbol):
            if symbol in ["^TNX", "^IRX"]:
                return {"last_price": 4.0}
            return {} # Fail others

        mock_instance.fetch_realtime_snapshot.side_effect = mock_snapshot
        mock_instance.fetch_market_data.return_value = {}

        # Run
        result = asyncio.run(self.agent.execute())

        # Verify Confidence < 1.0
        self.assertLess(result["confidence_score"], 1.0)
        # TNX and IRX worked, so 2/6 = 0.33 confidence
        self.assertAlmostEqual(result["confidence_score"], 2/6, delta=0.1)

if __name__ == '__main__':
    unittest.main()
