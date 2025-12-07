
from core.data_sources.yfinance_market_data import YFinanceMarketData
from core.utils.market_data_utils import format_market_data_gold_standard
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def fetch_and_save(symbol: str, output_dir: str):
    yf = YFinanceMarketData()
    print(f"Fetching data for {symbol}...")

    snapshot = yf.get_snapshot(symbol)
    intraday = yf.get_intraday_data(symbol)
    intra_year = yf.get_historical_data(symbol, period="1y")
    long_term = yf.get_long_term_data(symbol)

    formatted_data = format_market_data_gold_standard(symbol, snapshot, intraday, intra_year, long_term)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"{symbol.lower()}_market_data.json")
    with open(output_path, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    output_dir = "data/gold_standard"
    # Fetching SPY as a representative sample
    fetch_and_save("SPY", output_dir)
