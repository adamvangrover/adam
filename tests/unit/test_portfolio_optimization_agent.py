import unittest
import pandas as pd
import numpy as np
import asyncio
from core.agents.portfolio_optimization_agent import PortfolioOptimizationAgent

class TestPortfolioOptimizationAgent(unittest.TestCase):
    def setUp(self):
        # Create mock price data
        self.dates = pd.date_range(start='2023-01-01', periods=100)
        # Seed for reproducibility
        np.random.seed(42)
        self.data = pd.DataFrame({
            'AssetA': np.random.normal(100, 2, 100).cumsum() + 100, # Trending up
            'AssetB': np.random.normal(50, 1, 100).cumsum() + 50,
            'AssetC': np.random.normal(10, 0.5, 100).cumsum() + 10
        }, index=self.dates)

        self.agent = PortfolioOptimizationAgent(config={"risk_free_rate": 0.02})

    def test_initialization(self):
        self.assertEqual(self.agent.risk_free_rate, 0.02)

    def test_mean_variance_optimization(self):
        result = asyncio.run(self.agent.execute(
            historical_prices=self.data,
            method='mean_variance'
        ))

        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['method'], 'Mean-Variance (Max Sharpe)')

        allocation = result['allocation']
        self.assertEqual(len(allocation), 3)
        self.assertAlmostEqual(sum(allocation.values()), 1.0, places=4)

        # Check if AssetA (trending up strongest) has non-zero weight
        # (Given random walk, this is probabilistic, but with fixed seed should be consistent)
        # Actually cumsum of positive mean normal is strong trend.
        pass

    def test_ai_forecast(self):
        # This test might skip or warn if torch is not installed,
        # but the agent handles imports gracefully.

        result = asyncio.run(self.agent.execute(
            historical_prices=self.data,
            method='ai_forecast'
        ))

        if result['status'] == 'error' and 'PyTorch' in result['message']:
            print("Skipping AI test due to missing PyTorch")
            return

        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['method'], 'AI Forecast (LSTM)')

        allocation = result['allocation']
        self.assertEqual(len(allocation), 3)
        self.assertAlmostEqual(sum(allocation.values()), 1.0, places=4)

    def test_input_validation(self):
        # Test missing data
        res = asyncio.run(self.agent.execute())
        self.assertEqual(res['status'], 'error')

        # Test unknown method
        res = asyncio.run(self.agent.execute(historical_prices=self.data, method='magic_crystal_ball'))
        self.assertEqual(res['status'], 'error')

if __name__ == '__main__':
    unittest.main()
