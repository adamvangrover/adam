import unittest
import pandas as pd
import numpy as np
from core.v30_architecture.python_intelligence.agents.quantitative_analyst import QuantitativeAnalyst

class TestQuantHistorical(unittest.TestCase):
    def setUp(self):
        self.agent = QuantitativeAnalyst()

        # Create dummy data
        dates = pd.date_range(start="2023-01-01", periods=100)
        # Create an uptrend
        prices = np.linspace(100, 150, 100)

        self.df = pd.DataFrame({
            "Open": prices,
            "High": prices + 1,
            "Low": prices - 1,
            "Close": prices,
            "Volume": 1000
        }, index=dates)

    def test_analyze_historical_bullish(self):
        result = self.agent.analyze_historical("TEST", self.df)

        self.assertIn("conviction", result)
        self.assertIn("signal", result)
        self.assertIn("reasons", result)

        # In a perfect linear uptrend:
        # Price > SMA50 (+10)
        # MACD likely positive (Bullish Crossover potentially or sustained) (+15)
        # RSI might be high (Overbought) (-20)
        # Base 50 + 10 + 15 - 20 = 55 (Neutral/Bullish leaning)

        print(f"\nBullish Test Result: {result}")
        self.assertTrue(result['conviction'] > 0)
        self.assertTrue(result['conviction'] <= 100)

    def test_insufficient_data(self):
        small_df = self.df.head(10)
        result = self.agent.analyze_historical("TEST", small_df)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Insufficient data")

if __name__ == "__main__":
    unittest.main()
