
import asyncio
import time
from typing import List
from core.trading.hft.hft_engine import MarketTick
from core.data_sources.yfinance_market_data import YFinanceMarketData
import logging

logger = logging.getLogger(__name__)

class YFinanceMarketDataHandler:
    """
    Real-time market data feed using yfinance snapshots.
    Since yfinance is not a WebSocket, this polls periodically.
    Suitable for 'Snapshot' trading or slower HFT.
    """
    def __init__(self, symbols: List[str], queue: asyncio.Queue, poll_interval: float = 5.0):
        self.symbols = symbols
        self.queue = queue
        self.poll_interval = poll_interval
        self.running = False
        self.client = YFinanceMarketData()

    async def start(self):
        self.running = True
        logger.info(f"[YFinanceFeed] Starting polling for {self.symbols} every {self.poll_interval}s...")

        while self.running:
            start_time = time.time()

            for symbol in self.symbols:
                # Fetch snapshot
                snapshot = self.client.get_snapshot(symbol)

                if snapshot and snapshot.get('current_price'):
                    price = float(snapshot['current_price'])
                    # Estimate bid/ask if not provided (yfinance sometimes has them)
                    bid = float(snapshot.get('bid', price - 0.01))
                    ask = float(snapshot.get('ask', price + 0.01))

                    tick = MarketTick(
                        symbol=symbol,
                        bid=bid,
                        ask=ask,
                        timestamp=time.time()
                    )
                    await self.queue.put(tick)
                else:
                    logger.warning(f"[YFinanceFeed] No price data for {symbol}")

            # Sleep for the remainder of the interval
            elapsed = time.time() - start_time
            sleep_time = max(0.1, self.poll_interval - elapsed)
            await asyncio.sleep(sleep_time)

    def stop(self):
        self.running = False
