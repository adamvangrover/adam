import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
import pandas as pd
import numpy as np
from core.agents.quantum_portfolio_manager_agent import QuantumPortfolioManagerAgent
from core.schemas.agent_schema import AgentInput, AgentOutput

class TestQuantumPortfolioManagerAgent(unittest.TestCase):

    def setUp(self):
        self.config = {"lookback_period": "6mo"}
        # Do not instantiate here if we want to test with patched classes in individual tests
        # self.agent = QuantumPortfolioManagerAgent(self.config)

    @patch('core.agents.quantum_portfolio_manager_agent.QuantumMonteCarloBridge')
    @patch('core.agents.quantum_portfolio_manager_agent.DataFetcher')
    def test_execute_standard_flow(self, MockDataFetcher, MockBridge):
        mock_fetcher = MockDataFetcher.return_value
        mock_bridge = MockBridge.return_value

        # Mock historical data: 2 tickers, 5 days
        data_a = [
            {"date": "2023-01-01", "close": 100},
            {"date": "2023-01-02", "close": 101},
            {"date": "2023-01-03", "close": 102},
            {"date": "2023-01-04", "close": 103},
            {"date": "2023-01-05", "close": 104},
        ]
        data_b = [
            {"date": "2023-01-01", "close": 50},
            {"date": "2023-01-02", "close": 52},
            {"date": "2023-01-03", "close": 48},
            {"date": "2023-01-04", "close": 53},
            {"date": "2023-01-05", "close": 55},
        ]

        def fetch_side_effect(ticker, period, interval):
            if ticker == "A":
                return data_a
            elif ticker == "B":
                return data_b
            return []

        mock_fetcher.fetch_historical_data.side_effect = fetch_side_effect

        mock_bridge.optimize_portfolio.return_value = {
            "method": "QAOA",
            "allocation": {"A": 0.6, "B": 0.4},
            "energy": -1.23
        }

        # Instantiate with mocks active
        agent = QuantumPortfolioManagerAgent(self.config)

        # Use clean query to avoid parsing issues
        input_data = AgentInput(query="A, B", context={})

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(agent.execute(input_data))
        finally:
            loop.close()

        # Check type
        self.assertIsInstance(result, AgentOutput)
        self.assertEqual(result.metadata["status"], "success")
        self.assertEqual(result.metadata["tickers_processed"], ["A", "B"])
        self.assertEqual(result.metadata["optimized_allocation"]["A"], 0.6)

        # Verify calls
        self.assertEqual(mock_fetcher.fetch_historical_data.call_count, 2)
        mock_bridge.optimize_portfolio.assert_called_once()

        call_args = mock_bridge.optimize_portfolio.call_args
        # args[0] is (tickers, mu, sigma)
        self.assertEqual(call_args[0][0], ["A", "B"])
        self.assertIsInstance(call_args[0][1], np.ndarray)
        self.assertIsInstance(call_args[0][2], np.ndarray)

    @patch('core.agents.quantum_portfolio_manager_agent.QuantumMonteCarloBridge')
    @patch('core.agents.quantum_portfolio_manager_agent.DataFetcher')
    def test_execute_insufficient_data(self, MockDataFetcher, MockBridge):
        mock_fetcher = MockDataFetcher.return_value
        mock_fetcher.fetch_historical_data.return_value = [] # No data

        agent = QuantumPortfolioManagerAgent(self.config)

        # Provide a ticker that has no data, explicitly as dict input
        input_data = {"tickers": ["Z"]}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(agent.execute(input_data))
        finally:
            loop.close()

        # Returns dict on error
        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "failed")
        self.assertIn("Insufficient data", result["error"])

if __name__ == "__main__":
    unittest.main()
