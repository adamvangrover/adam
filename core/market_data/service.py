from typing import List, Dict, Any, Optional, Union
import random
import datetime
import time
import logging
from collections import OrderedDict

# -------------------------------------------------------------------------
# DEPENDENCY MANAGEMENT
# Progressive Enhancement: Load yfinance if available, else warn and degrade.
# -------------------------------------------------------------------------
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MarketDataService")

if not YFINANCE_AVAILABLE:
    logger.warning("yfinance library not found. Service operating in SYNTHETIC ONLY mode.")


class LRUCache:
    """
    Least Recently Used (LRU) Cache.
    
    Why: Financial APIs have strict rate limits.
    How: Keeps the 'capacity' most recent items. When full, drops the oldest 
    accessed item to make room for the new one.
    """
    def __init__(self, capacity: int = 100, ttl_seconds: int = 60):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        
        # TTL Check: Has the data expired?
        if (time.time() - timestamp) > self.ttl:
            self.cache.pop(key)
            return None
            
        # Refresh position (mark as recently used)
        self.cache.move_to_end(key)
        return value

    def put(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        
        self.cache[key] = (value, time.time())
        
        # Evict if over capacity
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class MarketDataService:
    """
    Centralized Market Data Ingestion Engine.
    
    Features:
    - Live Feed: yfinance (with retries)
    - Caching: LRU (In-Memory)
    - Fallback: Deterministic Synthetic Generation (Brownian Motion)
    """

    def __init__(self, use_synthetic_only: bool = False):
        self.use_synthetic_only = use_synthetic_only or (not YFINANCE_AVAILABLE)
        # Cache capacity: 200 symbols, expires every 30 seconds
        self.cache = LRUCache(capacity=200, ttl_seconds=30) 

    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Orchestrates the fetch flow: Cache -> Live API -> Synthetic Fallback.
        """
        symbol = symbol.upper()

        # 1. Check Cache
        cached_quote = self.cache.get(symbol)
        if cached_quote:
            logger.debug(f"CACHE HIT: {symbol}")
            return cached_quote

        # 2. Fetch from Live Source (if enabled)
        quote = None
        if not self.use_synthetic_only:
            quote = self._fetch_yfinance_with_retry(symbol)

        # 3. Fallback to Synthetic (if live failed or disabled)
        if not quote:
            if not self.use_synthetic_only:
                logger.warning(f"Live fetch failed for {symbol}. Generating synthetic data.")
            quote = self._generate_synthetic_quote(symbol)

        # 4. Update Cache
        self.cache.put(symbol, quote)
        return quote

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Fetches historical OHLCV candles. 
        Note: We do NOT cache history in this version due to memory size constraints.
        """
        if not self.use_synthetic_only:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                if not df.empty:
                    return df.reset_index().to_dict(orient="records")
            except Exception as e:
                logger.error(f"Historical fetch failed for {symbol}: {e}")

        # Fallback
        return self._generate_synthetic_history(symbol, start_date, end_date)

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------------------------------

    def _fetch_yfinance_with_retry(self, symbol: str, retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Fetches data with Exponential Backoff.
        
        Strategy:
        Attempt 1: Immediate
        Attempt 2: Wait 0.5s
        Attempt 3: Wait 1.0s
        Attempt 4: Wait 2.0s
        """
        for i in range(retries):
            try:
                ticker = yf.Ticker(symbol)
                # fast_info is much faster/reliable than .info
                info = ticker.fast_info 
                last_price = info.last_price

                if last_price is None:
                    raise ValueError("Received empty price data")

                return {
                    "symbol": symbol,
                    "bid": round(last_price * 0.9995, 2),
                    "ask": round(last_price * 1.0005, 2),
                    "last": round(last_price, 2),
                    "volume": info.last_volume,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "source": "yfinance"
                }
            except Exception as e:
                wait_time = (2 ** i) * 0.5
                logger.warning(f"Attempt {i+1} failed for {symbol}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        return None

    def _generate_synthetic_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Generates deterministic synthetic data based on symbol hash.
        Ensures consistency: asking for 'AAPL' twice yields similar synthetic numbers.
        """
        seed = sum(ord(c) for c in symbol)
        # Mix time into seed so prices 'move' every minute
        random.seed(seed + int(time.time() / 60)) 
        
        base_price = 100.0 + (seed % 500)
        noise = random.uniform(-0.5, 0.5)
        price = base_price + noise
        
        return {
            "symbol": symbol,
            "bid": round(price * 0.999, 2),
            "ask": round(price * 1.001, 2),
            "last": round(price, 2),
            "volume": random.randint(100000, 5000000),
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "synthetic_fallback"
        }

    def _generate_synthetic_history(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Generates a Random Walk (Brownian Motion) price path."""
        data = []
        try:
            current = datetime.date.fromisoformat(start_date)
            end = datetime.date.fromisoformat(end_date)
        except ValueError:
            current = datetime.date.today() - datetime.timedelta(days=30)
            end = datetime.date.today()

        # Seed price based on symbol text
        price = 100.0 + (sum(ord(c) for c in symbol) % 100)

        while current <= end:
            # Random Walk: Price t = Price t-1 + Random(Gaussian)
            change = random.gauss(0, 2.0) 
            price += change
            price = max(0.01, price) # Prevent negative prices
            
            data.append({
                "date": current.isoformat(),
                "open": round(price - random.random(), 2),
                "high": round(price + random.random(), 2),
                "low": round(price - random.random(), 2),
                "close": round(price, 2),
                "volume": random.randint(1000, 10000)
            })
            current += datetime.timedelta(days=1)
        
        return data
