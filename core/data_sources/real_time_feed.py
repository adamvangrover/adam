from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import random

class MarketFeed(ABC):
    """
    Abstract interface for a Real-Time Data Source.
    """
    @abstractmethod
    def fetch_price(self, symbol: str) -> Optional[float]:
        pass

    @abstractmethod
    def fetch_news(self, symbol: str) -> List[Dict[str, str]]:
        pass

class ManualFeed(MarketFeed):
    """
    Feed for manual user overrides.
    """
    def fetch_price(self, symbol: str) -> Optional[float]:
        # This feed is push-based, usually.
        return None

    def fetch_news(self, symbol: str) -> List[Dict[str, str]]:
        return []

class SimulationFeed(MarketFeed):
    """
    Fallback feed using random walk.
    """
    def __init__(self, base_price_map: Dict[str, float]):
        self.prices = base_price_map

    def fetch_price(self, symbol: str) -> Optional[float]:
        current = self.prices.get(symbol, 100.0)
        # Drift
        new_price = current * (1 + random.normalvariate(0, 0.01))
        self.prices[symbol] = new_price
        return new_price

    def fetch_news(self, symbol: str) -> List[Dict[str, str]]:
        return []

class MockSearchFeed(MarketFeed):
    """
    Simulates a Search Tool fetching 'Live' data.
    In a real scenario, this would wrap GoogleSearch/BingSearch tools.
    """
    def fetch_price(self, symbol: str) -> Optional[float]:
        # Logic to "search" - here we just return a plausible variation of a known baseline
        # or simulate a 'Search Hit'
        return None # Passive

    def search_for_update(self, query: str) -> Dict[str, Any]:
        """
        Simulates parsing a search result snippet.
        Input: "current price of AAPL"
        Output: {"AAPL": 155.20}
        """
        # Mock parsing logic
        if "AAPL" in query.upper(): return {"AAPL": 155.00 + random.random()}
        if "BTC" in query.upper(): return {"BTC-USD": 95000.00 + random.random() * 1000}
        return {}
