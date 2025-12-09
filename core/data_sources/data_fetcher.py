from __future__ import annotations
import yfinance as yf
import pandas as pd
import time
from typing import Dict, List, Any, Optional
from core.utils.logging_utils import get_logger

logger = get_logger(__name__)

class DataFetcher:
    """
    A robust data fetcher that retrieves live market data using yfinance.
    This class replaces manual JSON blobs with real-time data streaming.
    """

    def __init__(self):
        """Initialize the DataFetcher."""
        pass

    def fetch_market_data(self, ticker_symbol: str) -> Dict[str, Any]:
        """
        Fetches current market data for a given ticker.

        Args:
            ticker_symbol (str): The stock ticker symbol (e.g., "AAPL", "NVDA").

        Returns:
            Dict[str, Any]: A dictionary containing price, volume, and other market metrics.
                            Returns an empty dict if fetching fails.
        """
        try:
            logger.info(f"Fetching market data for {ticker_symbol}...")
            ticker = yf.Ticker(ticker_symbol)

            # .info is slower but provides rich fundamental data (sector, PE, etc.)
            # essential for the "Deep Dive" and "Market Mayhem" use cases.
            info = ticker.info

            if not info:
                logger.warning(f"No info found for {ticker_symbol}")
                return {}

            # Construct a clean dictionary with fallbacks
            data = {
                "symbol": ticker_symbol,
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "open": info.get("open") or info.get("regularMarketOpen"),
                "high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
                "low": info.get("dayLow") or info.get("regularMarketDayLow"),
                "volume": info.get("volume") or info.get("regularMarketVolume"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "dividend_yield": info.get("dividendYield"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "description": info.get("longBusinessSummary"),
                "currency": info.get("currency"),
                "exchange": info.get("exchange")
            }
            logger.info(f"Successfully fetched data for {ticker_symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching market data for {ticker_symbol}: {e}")
            return {}

    def fetch_historical_data(self, ticker_symbol: str, period: str = "1mo", interval: str = "1d") -> List[Dict[str, Any]]:
        """
        Fetches historical price data.

        Args:
            ticker_symbol (str): The stock ticker symbol.
            period (str): The data period to download.
            interval (str): The data interval.

        Returns:
            List[Dict[str, Any]]: A list of historical data points.
        """
        try:
            logger.info(f"Fetching historical data for {ticker_symbol} (period={period}, interval={interval})...")
            ticker = yf.Ticker(ticker_symbol)
            history: pd.DataFrame = ticker.history(period=period, interval=interval)

            if history.empty:
                logger.warning(f"No historical data found for {ticker_symbol}")
                return []

            results = []
            # Reset index to get Date/Datetime as a column
            history_reset = history.reset_index()

            for _, row in history_reset.iterrows():
                # Handle Date vs Datetime (yfinance uses 'Datetime' for intraday)
                date_val = row.get("Date")
                if date_val is None:
                    date_val = row.get("Datetime")

                # If still None, maybe index wasn't named?
                if date_val is None:
                    # Fallback to the first column if it looks like a date?
                    # Or just skip.
                    continue

                if hasattr(date_val, "isoformat"):
                    date_str = date_val.isoformat()
                else:
                    date_str = str(date_val)

                results.append({
                    "date": date_str,
                    "open": row.get("Open"),
                    "high": row.get("High"),
                    "low": row.get("Low"),
                    "close": row.get("Close"),
                    "volume": row.get("Volume"),
                    "dividends": row.get("Dividends"),
                    "stock_splits": row.get("Stock Splits")
                })

            logger.info(f"Successfully fetched {len(results)} historical records for {ticker_symbol}")
            return results

        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker_symbol}: {e}")
            return []

    def fetch_news(self, ticker_symbol: str) -> List[Dict[str, Any]]:
        """
        Fetches latest news for the ticker.

        Args:
            ticker_symbol (str): The stock ticker symbol.

        Returns:
            List[Dict[str, Any]]: A list of news items.
        """
        try:
            logger.info(f"Fetching news for {ticker_symbol}...")
            ticker = yf.Ticker(ticker_symbol)
            news = ticker.news

            if not news:
                logger.warning(f"No news found for {ticker_symbol}")
                return []

            results = []
            for item in news:
                results.append({
                    "title": item.get("title"),
                    "publisher": item.get("publisher"),
                    "link": item.get("link"),
                    "providerPublishTime": item.get("providerPublishTime"),
                    "type": item.get("type"),
                    "thumbnail": item.get("thumbnail")
                })

            logger.info(f"Successfully fetched {len(results)} news items for {ticker_symbol}")
            return results

        except Exception as e:
            logger.error(f"Error fetching news for {ticker_symbol}: {e}")
            return []

    def fetch_financials(self, ticker_symbol: str) -> Dict[str, Any]:
        """
        Fetches financial statements (balance sheet, income statement, cash flow).

        Args:
            ticker_symbol (str): The stock ticker symbol.

        Returns:
            Dict[str, Any]: A dictionary containing financial statements.
        """
        try:
            logger.info(f"Fetching financials for {ticker_symbol}...")
            ticker = yf.Ticker(ticker_symbol)

            balance_sheet = ticker.balance_sheet
            income_stmt = ticker.income_stmt
            cash_flow = ticker.cashflow

            def df_to_json_friendly(df: pd.DataFrame) -> Dict[str, Any]:
                if df.empty:
                    return {}
                result = {}
                for date_col in df.columns:
                    date_str = date_col.strftime('%Y-%m-%d') if hasattr(date_col, 'strftime') else str(date_col)
                    # Convert NaN to None for JSON compliance
                    col_data = df[date_col].where(pd.notnull(df[date_col]), None).to_dict()
                    result[date_str] = col_data
                return result

            data = {
                "balance_sheet": df_to_json_friendly(balance_sheet),
                "income_statement": df_to_json_friendly(income_stmt),
                "cash_flow": df_to_json_friendly(cash_flow)
            }

            logger.info(f"Successfully fetched financials for {ticker_symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching financials for {ticker_symbol}: {e}")
            return {}

    def fetch_realtime_snapshot(self, ticker_symbol: str) -> Dict[str, Any]:
        """
        Fetches a low-latency L1 market data snapshot using yfinance fast_info.
        """
        try:
            logger.info(f"Fetching realtime snapshot for {ticker_symbol}...")
            ticker = yf.Ticker(ticker_symbol)
            fi = ticker.fast_info

            # fast_info provides direct access to these properties
            snapshot = {
                "symbol": ticker_symbol,
                "last_price": fi.last_price,
                "previous_close": fi.previous_close,
                "open": fi.open,
                "day_high": fi.day_high,
                "day_low": fi.day_low,
                "year_high": fi.year_high,
                "year_low": fi.year_low,
                "market_cap": fi.market_cap,
                "currency": fi.currency,
                "timestamp": time.time()
            }
            logger.info(f"Successfully fetched snapshot for {ticker_symbol}")
            return snapshot
        except Exception as e:
            logger.error(f"Error fetching snapshot for {ticker_symbol}: {e}")
            return {}
