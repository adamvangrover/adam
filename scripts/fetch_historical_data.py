#!/usr/bin/env python3
import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.market_data.historical_loader import HistoricalLoader
from core.utils.logging_utils import get_logger

logger = get_logger("fetch_historical_data")

def main():
    logger.info("Starting historical data acquisition...")

    # Representative list of S&P 500 tickers across sectors
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", # Tech
        "JPM", "BAC", "V", "MA", "GS", # Finance
        "JNJ", "UNH", "PFE", "LLY", "ABBV", # Healthcare
        "XOM", "CVX", # Energy
        "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", # Consumer
        "BA", "CAT", "GE", # Industrial
        "T", "VZ" # Telecom
    ]

    loader = HistoricalLoader(data_dir="data/market_data")

    # We fetch from 1980 to 2025 as requested
    loader.run_pipeline(tickers, filename="sp500_history_1980_2025.parquet")

    logger.info("Historical data acquisition complete.")

if __name__ == "__main__":
    main()
