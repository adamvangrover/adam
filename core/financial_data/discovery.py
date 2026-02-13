import logging
from typing import List, Optional
try:
    import yfinance as yf
except ImportError:
    yf = None

from .schema import MarketTicker

logger = logging.getLogger(__name__)


class MarketDiscoveryAgent:
    """
    Agent responsible for the 'Discovery Layer' of the Financial Framework.
    Implements 'Real Search Pull' using yfinance to dynamically find assets.
    """

    def __init__(self):
        self.logger = logger
        if yf is None:
            self.logger.warning("yfinance is not installed. MarketDiscoveryAgent functionality will be limited.")

    def search_universe(self, query: str, limit: int = 20) -> List[MarketTicker]:
        """
        Searches the Yahoo Finance universe for assets matching the query.

        Args:
            query: The search string (e.g., "Technology", "ESG", "AAPL")
            limit: Maximum number of results to return

        Returns:
            List of validated MarketTicker objects.
        """
        if yf is None:
            self.logger.error("yfinance is not installed. Cannot perform search.")
            return []

        self.logger.info(f"Searching universe for: '{query}'")
        try:
            # yfinance Search returns a list of dictionaries in 'quotes'
            search_response = yf.Search(query, max_results=limit)
            raw_quotes = search_response.quotes

            tickers = []
            for quote in raw_quotes:
                try:
                    ticker = MarketTicker(**quote)
                    tickers.append(ticker)
                except Exception as e:
                    self.logger.warning(f"Failed to parse quote {quote.get('symbol')}: {e}")

            self.logger.info(f"Found {len(tickers)} results for '{query}'")
            return tickers

        except Exception as e:
            self.logger.error(f"Search failed for '{query}': {e}")
            return []

    def scan_sectors(self, sectors: List[str]) -> List[MarketTicker]:
        """
        Scans multiple sectors and aggregates results.
        """
        all_tickers = []
        for sector in sectors:
            results = self.search_universe(sector)
            all_tickers.extend(results)

        # Deduplicate by symbol
        unique_tickers = {t.symbol: t for t in all_tickers}
        return list(unique_tickers.values())
