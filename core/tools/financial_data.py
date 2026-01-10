import logging
from typing import Dict, Any, Optional
import datetime

logger = logging.getLogger(__name__)

class FinancialTools:
    """
    Registry of live financial tools.
    Implements the 'Check-then-Fetch' pattern.
    """

    def __init__(self):
        self.cache = {} # Simple in-memory cache

    def get_ticker_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetches basic price and volume data.
        """
        # 1. Check Cache (or Internal Knowledge Graph)
        if ticker in self.cache:
            logger.info(f"Cache hit for {ticker}")
            return self.cache[ticker]

        # 2. Fetch from External API
        # In a real scenario, use yfinance or alpha_vantage
        logger.info(f"Fetching live data for {ticker}...")
        try:
            # Mocking the external call
            data = self._mock_yfinance_fetch(ticker)
            self.cache[ticker] = data
            return data
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            return {"error": str(e)}

    def _mock_yfinance_fetch(self, ticker: str) -> Dict[str, Any]:
        """
        Simulates yfinance.Ticker(ticker).info
        """
        import random
        base_price = 100.0
        if ticker == "AAPL": base_price = 175.0
        if ticker == "TSLA": base_price = 240.0

        return {
            "symbol": ticker,
            "price": base_price + random.uniform(-5, 5),
            "volume": random.randint(1000000, 10000000),
            "timestamp": datetime.datetime.now().isoformat()
        }

    def search_news(self, query: str) -> List[Dict[str, Any]]:
        """
        Searches for recent news articles.
        """
        # 1. Check Internal KG for recent articles?
        # 2. Call Tavily/Serper
        logger.info(f"Searching news for: {query}")
        return [
            {
                "title": f"Market Update on {query}",
                "url": "https://finance.yahoo.com/...",
                "snippet": f"Analysts discuss the impact of recent events on {query}...",
                "source": "Bloomberg"
            }
        ]
