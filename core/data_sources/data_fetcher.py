from __future__ import annotations
try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import pandas as pd
except ImportError:
    pd = None

import time
from typing import Dict, List, Any, Optional
from core.utils.logging_utils import get_logger
from core.utils.retry_utils import retry_with_backoff

logger = get_logger(__name__)


class DataFetcher:
    """
    A robust data fetcher that retrieves live market data using yfinance.
    This class replaces manual JSON blobs with real-time data streaming.

    Serves as the primary 'Live Data Connector' for the Adam v23.5 system.
    """

    def __init__(self):
        """Initialize the DataFetcher."""
        pass

    @retry_with_backoff(retries=2, backoff_in_seconds=1)
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
                "previous_close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
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
                "exchange": info.get("exchange"),
                "country": info.get("country"),
                "debtToEquity": info.get("debtToEquity"),
                "profitMargins": info.get("profitMargins"),
                "returnOnEquity": info.get("returnOnEquity"),
                "totalCash": info.get("totalCash"),
                "totalDebt": info.get("totalDebt")
            }
            logger.info(f"Successfully fetched data for {ticker_symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching market data for {ticker_symbol}: {e}")
            return {}

    @retry_with_backoff(retries=2, backoff_in_seconds=1)
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

    @retry_with_backoff(retries=2, backoff_in_seconds=1)
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
                # yfinance news structure can be flat or nested in 'content'
                content = item.get("content", item)

                # Extract fields with fallbacks
                title = content.get("title")

                # Provider/Publisher extraction
                provider = content.get("provider", {})
                publisher = provider.get("displayName") if isinstance(provider, dict) else str(provider)

                # Link extraction
                click_through_url = content.get("clickThroughUrl")
                canonical_url = content.get("canonicalUrl")
                link = (click_through_url.get("url") if click_through_url else None) or \
                       (canonical_url.get("url") if canonical_url else None) or \
                    content.get("link")

                # Time extraction
                pub_date = content.get("pubDate") or content.get("providerPublishTime")

                results.append({
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "providerPublishTime": pub_date,
                    "type": content.get("contentType", content.get("type")),
                    "thumbnail": content.get("thumbnail")
                })

            logger.info(f"Successfully fetched {len(results)} news items for {ticker_symbol}")
            return results

        except Exception as e:
            logger.error(f"Error fetching news for {ticker_symbol}: {e}")
            return []

    @retry_with_backoff(retries=2, backoff_in_seconds=1)
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

            if yf is None:
                # Simulated Fallback if yfinance is missing
                import random
                price = 100.0 + random.uniform(-5, 5)
                return {
                    "symbol": ticker_symbol,
                    "last_price": price,
                    "previous_close": price * 0.99,
                    "open": price * 0.995,
                    "day_high": price * 1.01,
                    "day_low": price * 0.98,
                    "year_high": price * 1.2,
                    "year_low": price * 0.8,
                    "market_cap": 1_000_000_000,
                    "currency": "USD",
                    "timestamp": time.time(),
                    "simulated": True
                }

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

    @retry_with_backoff(retries=2, backoff_in_seconds=1)
    def fetch_recommendations(self, ticker_symbol: str) -> List[Dict[str, Any]]:
        """
        Fetches analyst recommendations.
        """
        try:
            logger.info(f"Fetching recommendations for {ticker_symbol}...")
            ticker = yf.Ticker(ticker_symbol)
            recs = ticker.recommendations
            if recs is None or recs.empty:
                return []

            # Handle DataFrame conversion
            if isinstance(recs, pd.DataFrame):
                # Reset index if it's not default range index, to ensure we get all data columns
                return recs.reset_index().to_dict(orient='records')

            return []
        except Exception as e:
            logger.error(f"Error fetching recommendations: {e}")
            return []

    @retry_with_backoff(retries=2, backoff_in_seconds=1)
    def fetch_calendar(self, ticker_symbol: str) -> Dict[str, Any]:
        """
        Fetches earnings calendar and other events.
        """
        try:
            logger.info(f"Fetching calendar for {ticker_symbol}...")
            ticker = yf.Ticker(ticker_symbol)
            cal = ticker.calendar
            if cal is None:
                return {}

            if isinstance(cal, pd.DataFrame):
                # Transpose to make it json friendly {Event: [Date]} or {0: {Earnings Date: ...}}
                return cal.to_dict()
            elif isinstance(cal, dict):
                return cal
            return {}
        except Exception as e:
            logger.error(f"Error fetching calendar: {e}")
            return {}

    @retry_with_backoff(retries=2, backoff_in_seconds=1)
    def fetch_credit_metrics(self) -> Dict[str, Any]:
        """
        Fetches credit market metrics.
        Specifically targeting High Yield (HYG) and Investment Grade (LQD) ETFs as proxies for CDX,
        and 10Y Treasury Yield (^TNX) to estimate spreads.
        """
        metrics = {}
        tickers = {
            "HYG": "High Yield Corp Bond ETF",
            "LQD": "Inv Grade Corp Bond ETF",
            "^TNX": "10-Year Treasury Yield"
        }

        for symbol, name in tickers.items():
            data = self.fetch_realtime_snapshot(symbol)
            if data and data.get("last_price"):
                metrics[symbol] = data
                metrics[symbol]['name'] = name
            else:
                 # Try to fetch standard market data if snapshot fails
                data = self.fetch_market_data(symbol)
                if data:
                     # Adapt to snapshot format lightly
                     metrics[symbol] = {
                         "symbol": symbol,
                         "last_price": data.get("current_price"),
                         "previous_close": data.get("previous_close"),
                         "name": name
                     }
        return metrics

    @retry_with_backoff(retries=2, backoff_in_seconds=1)
    def fetch_volatility_metrics(self) -> Dict[str, Any]:
        """
        Fetches volatility metrics: VIX, VIX3M, MOVE Index.
        """
        metrics = {}
        tickers = ["^VIX", "^VIX3M", "^MOVE"]

        for symbol in tickers:
            data = self.fetch_realtime_snapshot(symbol)
            if data and data.get("last_price"):
                 metrics[symbol] = data
            else:
                 # Fallback
                 data = self.fetch_market_data(symbol)
                 if data:
                     metrics[symbol] = {
                         "symbol": symbol,
                         "last_price": data.get("current_price"),
                         "previous_close": data.get("previous_close")
                     }
        return metrics

    @retry_with_backoff(retries=2, backoff_in_seconds=1)
    def fetch_crypto_metrics(self) -> Dict[str, Any]:
        """
        Fetches major crypto metrics: BTC-USD, ETH-USD.
        """
        metrics = {}
        tickers = ["BTC-USD", "ETH-USD"]

        for symbol in tickers:
            data = self.fetch_realtime_snapshot(symbol)
            if data and data.get("last_price"):
                 metrics[symbol] = data
            else:
                 data = self.fetch_market_data(symbol)
                 if data:
                     metrics[symbol] = {
                         "symbol": symbol,
                         "last_price": data.get("current_price"),
                         "previous_close": data.get("previous_close")
                     }
        return metrics

    @retry_with_backoff(retries=2, backoff_in_seconds=1)
    def fetch_treasury_metrics(self) -> Dict[str, Any]:
        """
        Fetches Treasury metrics for curve analysis.
        Uses ^TNX (10Y Yield), ^IRX (13W Yield), ^FVX (5Y Yield).
        Also fetches IEF (7-10 Year Treasury Bond ETF) for price action comparison.
        """
        metrics = {}
        tickers = {
            "^TNX": "10-Year Treasury Yield",
            "^IRX": "13-Week Treasury Bill",
            "^FVX": "5-Year Treasury Yield",
            "IEF": "iShares 7-10 Year Treasury Bond ETF"
        }

        for symbol, name in tickers.items():
            data = self.fetch_realtime_snapshot(symbol)
            if data and data.get("last_price"):
                 metrics[symbol] = data
                 metrics[symbol]['name'] = name
            else:
                 data = self.fetch_market_data(symbol)
                 if data:
                     metrics[symbol] = {
                         "symbol": symbol,
                         "last_price": data.get("current_price"),
                         "previous_close": data.get("previous_close"),
                         "name": name
                     }
        return metrics

    def fetch_macro_liquidity(self) -> Dict[str, Any]:
        """
        Fetches macro liquidity indicators.
        Note: True Fed Balance Sheet/TGA/Reverse Repo requires FRED data which is not available via yfinance.
        Returning a placeholder structure for the agent to use (or simulate).
        """
        # Placeholder for liquidity index
        return {
            "status": "simulated",
            "liquidity_index": 0.0, # Placeholder
            "components": {
                "fed_balance_sheet": 0.0,
                "tga": 0.0,
                "reverse_repo": 0.0
            },
            "note": "Real-time macro liquidity data requires FRED API integration."
        }
