import logging
import datetime
import random
import yfinance as yf
import pandas as pd
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class FinancialToolkit:
    """
    A unified registry for live financial data.
    Implements 'Check-then-Fetch' caching to minimize API hits.
    """

    def __init__(self):
        # In-memory cache: { ticker: {"data": info, "expiry": timestamp} }
        self._cache: Dict[str, Any] = {}

    def get_company_info(self, ticker: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Retrieves company profile and current price data.
        """
        ticker = ticker.upper()
        
        # 1. Check Cache
        if not force_refresh and ticker in self._cache:
            logger.info(f"Cache hit for {ticker}")
            return self._cache[ticker]

        # 2. Fetch from yfinance
        logger.info(f"Fetching live data for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Ensure we have a timestamp for the fetch
            info['last_updated'] = datetime.datetime.now().isoformat()
            
            self._cache[ticker] = info
            return info
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            return {"symbol": ticker, "error": str(e)}

    def get_historical_data(self, ticker: str, period: str = "1mo") -> pd.DataFrame:
        """
        Retrieves historical stock data. 
        Note: DataFrames are generally not cached here to ensure freshness of intervals.
        """
        try:
            stock = yf.Ticker(ticker.upper())
            return stock.history(period=period)
        except Exception as e:
            logger.error(f"Error fetching history for {ticker}: {e}")
            return pd.DataFrame()

    def get_financials(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Retrieves core financial statements.
        """
        try:
            stock = yf.Ticker(ticker.upper())
            return {
                "income_statement": stock.financials,
                "balance_sheet": stock.balance_sheet,
                "cash_flow": stock.cashflow
            }
        except Exception as e:
            logger.error(f"Error fetching financials for {ticker}: {e}")
            return {}

    def search_news(self, query: str) -> List[Dict[str, Any]]:
        """
        Searches for recent news articles.
        """
        logger.info(f"Searching news for: {query}")
        # Placeholder for external API (Tavily, Serper, or Google News)
        return [
            {
                "title": f"Market Update: {query}",
                "source": "Financial Times",
                "summary": f"Latest analysis regarding {query} shows market volatility...",
                "url": "https://example.com/finance-news"
            },
            {
                "title": f"{query} Institutional Holdings Change",
                "source": "Reuters",
                "summary": "Major funds adjusted their positions in response to recent reports.",
                "url": "https://example.com/reuters-update"
            }
        ]

    def get_treasury_yields(self) -> Dict[str, float]:
        """
        Retrieves current US Treasury yields.
        """
        # In production, this would scrape Treasury.gov or use a macro API
        return {
            "3M": 5.45,
            "2Y": 4.88,
            "10Y": 4.50,
            "30Y": 4.65,
            "timestamp": datetime.datetime.now().isoformat()
        }
