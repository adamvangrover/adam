"""
Discovery Layer for the Gold Standard Toolkit.
Implements Real Search Pull and Dynamic Universe Discovery.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List

try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)

class DiscoveryAgent:
    def __init__(self, storage_path: str = "data/universe"):
        self.storage_path = storage_path
        self.snapshots_path = os.path.join(storage_path, "snapshots")
        os.makedirs(self.snapshots_path, exist_ok=True)

    def search_assets(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Performs a semantic/keyword search using yfinance.
        Returns a list of quote objects (Ticker, Name, Type).
        """
        if yf is None:
            logger.error("yfinance not installed.")
            return []

        try:
            search_results = yf.Search(query, max_results=limit)
            # search_results.quotes is a list of dicts
            return search_results.quotes
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []

    def get_sector_universe(self, sector_key: str) -> List[str]:
        """
        Retrieves top companies/ETFs for a given sector.
        Note: This relies on yfinance Sector module.
        """
        if yf is None: return []

        try:
            sector = yf.Sector(sector_key)
            # Assuming sector.top_companies returns a dataframe or list of tickers
            # The API details might vary, creating a best-effort implementation
            return sector.top_companies.index.tolist() if hasattr(sector, 'top_companies') else []
        except Exception as e:
            logger.warning(f"Sector lookup failed for '{sector_key}': {e}")
            return []

    def get_industry_universe(self, industry_key: str) -> List[str]:
        """
        Retrieves companies for a specific industry.
        """
        if yf is None: return []

        try:
            industry = yf.Industry(industry_key)
            return industry.top_companies.index.tolist() if hasattr(industry, 'top_companies') else []
        except Exception as e:
            logger.warning(f"Industry lookup failed for '{industry_key}': {e}")
            return []

    def snapshot_universe(self, universe_name: str, tickers: List[str]):
        """
        Saves the current list of tickers for a universe to a JSON file.
        This creates a 'Historical Universe Snapshot' to mitigate survivorship bias.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d')
        filename = f"{universe_name}_{timestamp}.json"
        filepath = os.path.join(self.snapshots_path, filename)

        data = {
            "universe_name": universe_name,
            "date": timestamp,
            "count": len(tickers),
            "tickers": tickers
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved universe snapshot: {filepath}")

    def run_discovery_cycle(self):
        """
        Example of an automated discovery cycle.
        """
        logger.info("Starting Discovery Cycle...")

        # Example: Search for new themes
        themes = ["Artificial Intelligence", "Quantum Computing", "Biotechnology"]
        for theme in themes:
            logger.info(f"Searching for {theme}...")
            results = self.search_assets(theme)
            tickers = [q['symbol'] for q in results if 'symbol' in q]
            self.snapshot_universe(f"theme_{theme.replace(' ', '_')}", tickers)

        # Example: Sector snapshots (keys depend on Yahoo's taxonomy)
        # self.snapshot_universe("sector_tech", self.get_sector_universe("technology"))

        logger.info("Discovery Cycle Complete.")
