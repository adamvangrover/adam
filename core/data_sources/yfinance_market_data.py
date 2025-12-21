
import yfinance as yf
import pandas as pd
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class YFinanceMarketData:
    """
    A data source wrapper for fetching market data via yfinance.
    Provides intra-day, intra-year, and long-term data.
    """

    def __init__(self):
        pass

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches a current snapshot of the market data for the symbol.
        This includes current price, basic fundamentals, etc.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            # Provide a safe fallback if info is empty or keys are missing
            if not info:
                logger.warning(f"No info found for symbol {symbol}")
                return {}

            snapshot = {
                "symbol": symbol,
                "current_price": info.get("currentPrice"),
                "open": info.get("open"),
                "high": info.get("dayHigh"),
                "low": info.get("dayLow"),
                "volume": info.get("volume"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "dividend_yield": info.get("dividendYield"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "long_business_summary": info.get("longBusinessSummary"),
            }
            return snapshot
        except Exception as e:
            logger.error(f"Error fetching snapshot for {symbol}: {e}")
            return {}

    def get_intraday_data(self, symbol: str, interval: str = "1m", period: str = "1d") -> List[Dict[str, Any]]:
        """
        Fetches intraday data.
        """
        try:
            ticker = yf.Ticker(symbol)
            # Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
            df = ticker.history(period=period, interval=interval)

            data = []
            for index, row in df.iterrows():
                data.append({
                    "timestamp": index.isoformat(),
                    "open": row["Open"],
                    "high": row["High"],
                    "low": row["Low"],
                    "close": row["Close"],
                    "volume": row["Volume"]
                })
            return data
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return []

    def get_historical_data(self, symbol: str, period: str = "1y") -> List[Dict[str, Any]]:
        """
        Fetches historical data (intra-year).
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            data = []
            for index, row in df.iterrows():
                # index is Date (usually just date, not datetime for daily)
                # But yfinance returns datetime index even for daily
                data.append({
                    "date": index.strftime('%Y-%m-%d'),
                    "open": row["Open"],
                    "high": row["High"],
                    "low": row["Low"],
                    "close": row["Close"],
                    "volume": row["Volume"]
                })
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []

    def get_long_term_data(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetches long-term data (max history, monthly interval).
        For Robo Advisor analysis.
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="max", interval="1mo")

            data = []
            for index, row in df.iterrows():
                data.append({
                    "date": index.strftime('%Y-%m-%d'),
                    "open": row["Open"],
                    "high": row["High"],
                    "low": row["Low"],
                    "close": row["Close"],
                    "volume": row["Volume"]
                })
            return data
        except Exception as e:
            logger.error(f"Error fetching long-term data for {symbol}: {e}")
            return []
