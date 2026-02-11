import os
import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

DATA_FILE = "data/historical_market_data.json"
TICKERS = ["SPY", "QQQ", "IWM", "BTC-USD", "^VIX", "^TNX"]

def fetch_and_cache_data():
    """
    Fetches historical data from yfinance and caches it to a JSON file.
    """
    print(f"Fetching historical data for {TICKERS}...")

    # Define range: Start of 2024 to present (plus some buffer for future simulation base)
    start_date = "2024-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        # Download data
        data = yf.download(TICKERS, start=start_date, end=end_date, group_by='ticker')

        # Structure for JSON
        # { "YYYY-MM-DD": { "SPY": { "Close": 123.45, ... }, ... } }
        formatted_data = {}

        # Iterate through dates
        # Note: yfinance returns a MultiIndex DataFrame if multiple tickers
        # Level 0: Ticker, Level 1: OHLCV

        # Normalize index to string dates
        data.index = data.index.strftime('%Y-%m-%d')

        for date_str, row in data.iterrows():
            formatted_data[date_str] = {}
            for ticker in TICKERS:
                try:
                    # Accessing MultiIndex can be tricky safely
                    if ticker in data.columns.get_level_values(0):
                        ticker_data = data[ticker].loc[date_str]
                        # Check for NaN (trading holidays etc)
                        if not pd.isna(ticker_data['Close']):
                            formatted_data[date_str][ticker] = {
                                "Open": float(ticker_data['Open']),
                                "High": float(ticker_data['High']),
                                "Low": float(ticker_data['Low']),
                                "Close": float(ticker_data['Close']),
                                "Volume": int(ticker_data['Volume'])
                            }
                except Exception as e:
                    # Ticker might not have data for this day
                    pass

        # Ensure directory exists
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

        with open(DATA_FILE, "w") as f:
            json.dump(formatted_data, f, indent=2)

        print(f"Successfully cached {len(formatted_data)} days of market data to {DATA_FILE}")
        return formatted_data

    except Exception as e:
        print(f"Error fetching data: {e}")
        return {}

def load_data():
    """Loads data from cache, or fetches if missing."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    else:
        return fetch_and_cache_data()

if __name__ == "__main__":
    fetch_and_cache_data()
