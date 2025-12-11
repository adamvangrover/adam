"""
Data Cleaning Module for Intra-Day Trading.
Handles zero volume, missing bars, and imputation.
"""

import pandas as pd
import numpy as np

def clean_intraday_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw 1-minute data for trading.
    1. Resamples to 1min to enforce continuous time grid.
    2. Forward fills prices (flatline).
    3. Fills volume with 0.
    """
    if df.empty:
        return df

    # Ensure index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    # Resample to ensure every minute exists
    # We use a custom aggregation to handle the resampling logic
    # Note: 'first', 'max' etc. on NaNs return NaNs.
    df_resampled = df.resample('1min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    # Imputation Strategy
    # Forward fill Close price first (price continuity)
    df_resampled['Close'] = df_resampled['Close'].ffill()

    # If Open/High/Low are missing, it means no trades occurred in that minute.
    # Set them equal to the Close price (flat candle).
    df_resampled['Open'] = df_resampled['Open'].fillna(df_resampled['Close'])
    df_resampled['High'] = df_resampled['High'].fillna(df_resampled['Close'])
    df_resampled['Low'] = df_resampled['Low'].fillna(df_resampled['Close'])

    # Fill missing Volume with 0
    df_resampled['Volume'] = df_resampled['Volume'].fillna(0)

    return df_resampled
