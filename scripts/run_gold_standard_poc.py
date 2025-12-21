"""
POC Script for the Gold Standard Financial Toolkit.
Demonstrates the storage, retrieval, and analysis pipeline using mock data.
"""

import os
import sys

import numpy as np
import pandas as pd

# Ensure core is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from core.gold_standard.advisory.mpt import PortfolioOptimizer
    from core.gold_standard.qa import validate_dataframe
    from core.gold_standard.storage import StorageEngine
    from core.gold_standard.trading.strategy import MeanReversionStrategy
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def main():
    print("=== Adam Gold Standard Toolkit POC ===")

    # Initialize Storage
    storage_path = "data/gold_standard_poc"
    storage = StorageEngine(base_path=storage_path)

    # --- 1. Simulate Daily Data Ingestion ---
    print("\n[1] Simulating Daily Data Ingestion...")
    dates = pd.date_range(start="2023-01-01", periods=100, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    data_frames = []
    for t in tickers:
        # Generate random price walk
        price_path = 100 + np.random.randn(100).cumsum()
        df = pd.DataFrame({
            'Open': price_path,
            'High': price_path + 1,
            'Low': price_path - 1,
            'Close': price_path,
            'Volume': np.random.randint(1000, 10000, 100),
            'Ticker': t
        }, index=dates)
        data_frames.append(df)

    full_daily_df = pd.concat(data_frames)

    try:
        storage.store_daily(full_daily_df)
        print("Daily data stored successfully.")
    except Exception as e:
        print(f"Storage failed (check PyArrow installation): {e}")

    # --- 2. Load Daily Data & Run MPT ---
    print("\n[2] Loading Daily Data & Running MPT...")
    try:
        loaded_df = storage.load_daily()
        print(f"Loaded {len(loaded_df)} rows from storage.")

        # Pivot for MPT (Date x Ticker)
        # Note: loaded_df might have duplicate index due to multiple tickers.
        # We need to pivot.
        if 'Ticker' in loaded_df.columns:
            prices = loaded_df.pivot(columns='Ticker', values='Close')

            # Initialize Optimizer
            opt = PortfolioOptimizer(prices)
            weights = opt.optimize_max_sharpe()
            if weights:
                print("Optimized Portfolio Weights (Max Sharpe):")
                for k, v in weights.items():
                    print(f"  {k}: {v:.4f}")

                metrics = opt.calculate_risk_metrics(weights)
                print("Risk Metrics:")
                for k, v in metrics.items():
                    print(f"  {k}: {v:.4f}")
            else:
                print("Optimization returned empty weights (likely missing PyPortfolioOpt).")
        else:
            print("Loaded data does not contain 'Ticker' column.")

    except Exception as e:
        print(f"MPT Analysis failed: {e}")

    # --- 3. Simulate Intraday Data & Run Trading Strategy ---
    print("\n[3] Simulating Intraday Data & Trading Strategy...")
    intraday_dates = pd.date_range(start="2023-10-27 09:30", periods=60, freq='1min')
    price_path = 150 + np.random.randn(60).cumsum()

    intraday_df = pd.DataFrame({
        'Open': price_path,
        'High': price_path + 0.5,
        'Low': price_path - 0.5,
        'Close': price_path,
        'Volume': np.random.randint(100, 500, 60)
    }, index=intraday_dates)

    # Run Strategy
    strat = MeanReversionStrategy(window=10, z_threshold=1.5)
    signals_df = strat.generate_signals(intraday_df)

    print("Generated Signals (Last 5 mins):")
    print(signals_df[['Close', 'Z_Score', 'Signal']].tail())

    # Store Intraday
    try:
        storage.store_intraday(intraday_df, ticker="TSLA")
        print("Intraday data stored successfully.")
    except Exception as e:
        print(f"Intraday storage failed: {e}")

    print("\n=== POC Complete ===")

if __name__ == "__main__":
    main()
