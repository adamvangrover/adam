"""
Mean Reversion Strategy for Intra-Day Trading.
"""

import pandas as pd
import numpy as np


class MeanReversionStrategy:
    def __init__(self, window: int = 20, z_threshold: float = 2.0):
        self.window = window
        self.z_threshold = z_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Z-Score and generates trading signals.
        Input df must have a 'Close' column.
        """
        if df.empty:
            return pd.DataFrame()

        df = df.copy()

        # Calculate Rolling Stats (Mu and Sigma)
        df['SMA'] = df['Close'].rolling(window=self.window).mean()
        df['StdDev'] = df['Close'].rolling(window=self.window).std()

        # Calculate Z-Score: (Price - Mean) / StdDev
        # Avoid division by zero
        df['Z_Score'] = (df['Close'] - df['SMA']) / df['StdDev'].replace(0, np.nan)

        # Signal Logic
        # 1 = Long (Oversold)
        # -1 = Short (Overbought)
        # 0 = Neutral/Exit

        df['Signal'] = 0

        # Entry Long: Z < -2.0
        df.loc[df['Z_Score'] < -self.z_threshold, 'Signal'] = 1

        # Entry Short: Z > 2.0
        df.loc[df['Z_Score'] > self.z_threshold, 'Signal'] = -1

        # Note: A real system needs state management (Holding, Cash, etc.)
        # This returns raw signal generation based on the bar.

        return df
