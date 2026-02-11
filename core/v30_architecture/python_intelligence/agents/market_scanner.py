import asyncio
import logging
import yfinance as yf
import pandas as pd
import random
from typing import List

# Import BaseAgent
try:
    from core.v30_architecture.python_intelligence.agents.base_agent import BaseAgent
except ImportError:
    # If run as script
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
    from core.v30_architecture.python_intelligence.agents.base_agent import BaseAgent

logger = logging.getLogger("MarketScanner")

class MarketScanner(BaseAgent):
    """
    Real-time Market Scanner Agent.
    Fetches live market data using yfinance and emits market_data events.
    Replaces the simulated random data generator.
    """
    def __init__(self, tickers: List[str] = None):
        super().__init__("MarketScanner-V2", "data_acquisition")
        if tickers is None:
            self.tickers = ["BTC-USD", "SPY", "QQQ", "IWM", "ETH-USD", "VIX", "MSFT", "NVDA", "AAPL"]
        else:
            self.tickers = tickers

    async def run(self):
        logger.info(f"{self.name} started. Scanning: {self.tickers}")
        while True:
            try:
                # Fetch data in a thread to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                data = await loop.run_in_executor(None, self._fetch_batch_data)

                if data is not None and not data.empty:
                    await self._process_and_emit(data)

            except Exception as e:
                logger.error(f"Error in MarketScanner loop: {e}")

            # Sleep for a random interval (e.g., 60-120 seconds to avoid rate limits)
            await asyncio.sleep(random.uniform(60.0, 120.0))

    def _fetch_batch_data(self):
        try:
            # Fetch 5 days of 15m data to get recent price and calculate change
            # period="5d", interval="15m" is a good balance for intraday updates
            df = yf.download(self.tickers, period="5d", interval="15m", group_by='ticker', progress=False, threads=True)
            return df
        except Exception as e:
            logger.error(f"yfinance batch fetch error: {e}")
            return None

    async def _process_and_emit(self, data: pd.DataFrame):
        for symbol in self.tickers:
            try:
                # Extract DF for symbol
                df = pd.DataFrame()
                if isinstance(data.columns, pd.MultiIndex):
                    try:
                        if symbol in data.columns:
                            df = data[symbol]
                    except KeyError:
                        continue
                else:
                    # Single ticker or flat structure
                    if len(self.tickers) == 1:
                        df = data
                    elif 'Close' in data.columns:
                        # Fallback if structure is flat but we have multiple tickers (rare with group_by)
                        # We might need to check if columns have ticker name level
                        pass

                if df.empty or 'Close' not in df.columns or len(df) < 1:
                    continue

                # Drop NaNs
                df = df.dropna(subset=['Close'])
                if df.empty:
                    continue

                # Get latest close
                current_price = df['Close'].iloc[-1]
                volume = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0

                # Calculate change pct
                change_pct = 0.0
                # We need the open of the day or previous close.
                # Since we have 15m data, we can look back.
                # Simple approximation: Compare to 24h ago (approx 96 periods) or previous close?
                # For "Change Pct", usually it means "Since Previous Daily Close".
                # But with 15m data, we only have intraday bars.
                # Let's approximate using the first bar of the current day if possible, or just the previous bar for "momentum"?
                # Standard is % change from previous day close.
                # But here I'll use change from the start of the fetched window or previous bar?
                # Let's use % change from the previous bar (immediate momentum) or
                # better yet, let's fetch daily data separately? No, too many requests.
                # Let's use the % change of the last bar itself (Close vs Open) or
                # % change from 1 bar ago.
                # The simulated scanner did: random change.
                # Let's do: (Current Price - Price 1 hour ago) / Price 1 hour ago * 100
                if len(df) >= 5: # 15m * 4 = 1h
                    prev_price = df['Close'].iloc[-5]
                    if prev_price > 0:
                        change_pct = ((current_price - prev_price) / prev_price) * 100
                elif len(df) >= 2:
                     prev_price = df['Close'].iloc[-2]
                     if prev_price > 0:
                        change_pct = ((current_price - prev_price) / prev_price) * 100

                # Emit
                payload = {
                    "symbol": symbol,
                    "price": round(float(current_price), 2),
                    "change_pct": round(float(change_pct), 2),
                    "volume": int(volume)
                }

                await self.emit("market_data", payload)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
