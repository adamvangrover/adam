from typing import List, Dict, Any, Optional
import random
import datetime

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

class MarketDataService:
    """
    Centralized service for ingesting and normalizing market data.
    Supports real-time feeds (via yfinance if available) and robust synthetic fallback.
    """

    def __init__(self, sources: List[str] = None):
        self.sources = sources or ["yfinance" if YFINANCE_AVAILABLE else "synthetic"]

    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch the latest quote for a symbol."""
        if YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                # Fast retrieval
                info = ticker.fast_info
                return {
                    "symbol": symbol,
                    "bid": info.last_price * 0.9995,
                    "ask": info.last_price * 1.0005,
                    "volume": info.last_volume,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "source": "yfinance"
                }
            except Exception as e:
                print(f"yfinance failed for {symbol}: {e}")
                # Fallthrough to synthetic

        # Synthetic Fallback
        base_price = 100.0 + (sum(ord(c) for c in symbol) % 100)
        return {
            "symbol": symbol,
            "bid": round(base_price * 0.999, 2),
            "ask": round(base_price * 1.001, 2),
            "volume": random.randint(100000, 5000000),
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "synthetic_fallback"
        }

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch historical data bars."""
        if YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                return df.reset_index().to_dict(orient="records")
            except Exception:
                pass

        # Synthetic History
        data = []
        current_date = datetime.date.fromisoformat(start_date)
        end = datetime.date.fromisoformat(end_date)
        price = 100.0

        while current_date <= end:
            change = random.uniform(-2.0, 2.0)
            price += change
            data.append({
                "date": current_date.isoformat(),
                "open": price - random.random(),
                "high": price + random.random(),
                "low": price - random.random(),
                "close": price,
                "volume": random.randint(1000, 10000)
            })
            current_date += datetime.timedelta(days=1)

        return data
