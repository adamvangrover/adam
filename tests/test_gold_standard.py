from core.gold_standard.qa import validate_dataframe
from core.gold_standard.trading.strategy import MeanReversionStrategy
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Ensure importability
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestGoldStandard(unittest.TestCase):
    def test_mean_reversion(self):
        # Create enough data for a window of 5
        data = [100] * 20
        data[10] = 110  # Spike
        df = pd.DataFrame({'Close': data})

        strat = MeanReversionStrategy(window=5, z_threshold=1.5)
        res = strat.generate_signals(df)

        self.assertIn('Z_Score', res.columns)
        self.assertIn('Signal', res.columns)

        # Check that we got a signal (spike should trigger Short/-1)
        # Row 10 is the spike. SMA of prev 5 is 100. Val is 110. Z ~ high.
        # But rolling is backward looking including current? usually.
        # pandas rolling includes current.
        # So mean=(100*4 + 110)/5 = 102. Std dev exists.
        # (110 - 102) / std > threshold probably.

        self.assertFalse(res['Signal'].abs().max() == 0, "Should generate some signals")

    def test_qa_validation(self):
        try:
            import pandera
        except ImportError:
            self.skipTest("Pandera not installed")

        df = pd.DataFrame({
            'Open': [100.0], 'High': [105.0], 'Low': [95.0], 'Close': [100.0], 'Volume': [1000.0]
        }, index=pd.to_datetime(['2023-01-01']))
        df.index.name = "Date"

        # Should pass
        try:
            validate_dataframe(df)
        except Exception as e:
            self.fail(f"Validation raised validation error: {e}")

        # Should fail (negative price)
        bad_df = df.copy()
        bad_df.iloc[0, 0] = -5.0  # Open

        with self.assertRaises(Exception):
            validate_dataframe(bad_df)


if __name__ == '__main__':
    unittest.main()
