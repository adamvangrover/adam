"""
Module 1: High-Frequency Trading (HFT) Execution Engine Architecture
====================================================================
Architectural Paradigm: Asynchronous Event-Driven Design via Python asyncio.

Components:
- Market Data Handler (The Producer): Simulates WebSocket ingestion.
- Strategy Engine (The Consumer): Market Making logic.
- Order Manager: State machine for order lifecycle.
- Circuit Breaker (The Risk Gate): Synchronous guardrail on critical path.

See "Master Architect" prompt output for details.
"""

import asyncio
import time
import logging
import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List

# Architect Note: Using dataclasses reduces memory overhead compared to dictionaries.
# 'slots=True' (in Python 3.10+) would further optimize attribute access speed.


@dataclass
class Tick:
    symbol: str
    bid: float
    ask: float
    timestamp: int  # Unix timestamp in milliseconds


class SystemState(Enum):
    RUNNING = 1
    HALTED_LATENCY = 2
    HALTED_DRAWDOWN = 3
    HALTED_MANUAL = 4


class CircuitBreaker:
    """
    The Risk Gate: Monitors system health metrics synchronously on the critical path.
    """

    def __init__(self, max_latency_ms: int = 50, max_drawdown_pct: float = 0.02):
        self.max_latency_ms = max_latency_ms
        self.max_drawdown = max_drawdown_pct
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.state = SystemState.RUNNING
        self.logger = logging.getLogger("RiskGate")

    def update_pnl(self, current_equity: float):
        """Updates the high-water mark and checks for drawdown violations."""
        if self.state != SystemState.RUNNING:
            return

        # Initialize peak equity on first run
        if self.peak_equity == 0.0:
            self.peak_equity = current_equity

        # Update High-Water Mark
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        self.current_equity = current_equity
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity

        if drawdown > self.max_drawdown:
            self.state = SystemState.HALTED_DRAWDOWN
            self.logger.critical(f"HALT: Max Drawdown breached. DD: {drawdown:.2%}, Limit: {self.max_drawdown:.2%}")
            raise SystemExit("Risk Limit Breached: Drawdown")

    def check_latency(self, exchange_ts_ms: int):
        """
        Checks if the incoming tick data is fresh.
        """
        if self.state != SystemState.RUNNING:
            return

        local_ts_ms = int(time.time() * 1000)
        latency = local_ts_ms - exchange_ts_ms

        if latency > self.max_latency_ms:
            self.state = SystemState.HALTED_LATENCY
            self.logger.critical(
                f"HALT: Latency threshold exceeded. Delta: {latency}ms, Limit: {self.max_latency_ms}ms")
            raise SystemExit("Risk Limit Breached: Latency")


class MarketDataHandler:
    """
    Asynchronous WebSocket consumer using the Producer pattern.
    """

    def __init__(self, uri: str, queue: asyncio.Queue, symbols: List[str]):
        self.uri = uri
        self.queue = queue
        self.symbols = symbols
        self.logger = logging.getLogger("MDHandler")

    async def connect(self):
        # Simulated WebSocket connection loop
        self.logger.info(f"Connecting to Exchange Stream: {self.uri}")
        while True:
            try:
                # In production: async with websockets.connect(self.uri) as ws:
                # await ws.send(json.dumps({"op": "subscribe", "args": self.symbols}))
                # async for message in ws:

                # SIMULATION OF TICK ARRIVAL
                await asyncio.sleep(0.01)  # Simulate 100 ticks/sec

                now_ms = int(time.time() * 1000)
                # Simulate a random latency spike occasionally
                if random.random() > 0.99:
                    jitter = random.randint(50, 100)  # Introduce artificial lag
                    now_ms -= jitter

                tick = Tick(
                    symbol="BTC-USD",
                    bid=50000.00,
                    ask=50005.00,
                    timestamp=now_ms
                )

                # Non-blocking put to the queue
                # If queue is full, this would block (backpressure), signaling strategy is too slow.
                await self.queue.put(tick)

            except Exception as e:
                self.logger.error(f"WebSocket Connection Error: {e}")
                await asyncio.sleep(5)  # Exponential backoff in real impl


class OrderManager:
    """
    Manages the lifecycle of orders (Place, Modify, Cancel).
    Abstracts the API complexity from the strategy.
    """

    def __init__(self):
        self.open_orders = {}  # Map
        self.inventory = 0.0
        self.logger = logging.getLogger("OrderManager")

    async def place_order(self, symbol: str, side: str, price: float, qty: float):
        """
        Sends order to exchange via REST or FIX.
        """
        # Simulate API Latency
        # await exchange_api.post_order(...)
        order_id = f"{side}-{int(time.time()*1000000)}"
        self.open_orders[order_id] = {'symbol': symbol, 'side': side, 'price': price, 'qty': qty}
        self.logger.info(f"PLACED {side.upper()} {qty} @ {price:.2f} | ID: {order_id}")
        return order_id

    async def cancel_all(self):
        """
        Safety routine to clear the book.
        """
        self.logger.warning("CANCELLING ALL ORDERS")
        # In production: await exchange_api.batch_cancel(self.open_orders.keys())
        self.open_orders.clear()


class MarketMakerStrategy:
    """
    The Core Logic Consumer.
    """

    def __init__(self, queue: asyncio.Queue, risk_gate: CircuitBreaker, om: OrderManager):
        self.queue = queue
        self.risk_gate = risk_gate
        self.om = om
        self.spread_bps = 5  # 5 basis points target spread
        self.position_limit = 1.0  # Max 1 BTC

    async def run(self):
        self.risk_gate.logger.info("Strategy Engine Active")
        while True:
            # 1. Wait for Tick (Yields control to Event Loop if empty)
            tick = await self.queue.get()

            try:
                # 2. Synchronous Risk Check (Critical Path)
                self.risk_gate.check_latency(tick.timestamp)

                # In a real system, we would also update PnL here based on
                # a separate execution feed or account balance stream.
                # self.risk_gate.update_pnl(current_nav)

                # 3. Strategy Logic
                mid_price = (tick.bid + tick.ask) / 2.0

                # Calculate Spread Width
                half_spread = mid_price * (self.spread_bps / 10000.0)

                # Basic Inventory Skew Logic
                # If we are long, skew prices down to sell. If short, skew up to buy.
                skew = - (self.om.inventory / self.position_limit) * half_spread

                bid_price = mid_price - half_spread + skew
                ask_price = mid_price + half_spread + skew

                # 4. Execution Logic
                # Cancel previous orders (Simplified: usually we modify/replace)
                await self.om.cancel_all()

                # Place new ladder
                # Use asyncio.gather to send both requests concurrently
                await asyncio.gather(
                    self.om.place_order(tick.symbol, 'buy', bid_price, 0.01),
                    self.om.place_order(tick.symbol, 'sell', ask_price, 0.01)
                )

            except SystemExit:
                # Graceful shutdown on Risk Trip
                await self.om.cancel_all()
                self.risk_gate.logger.critical("SYSTEM HALTED BY CIRCUIT BREAKER")
                return  # Exit the loop

            except Exception as e:
                self.risk_gate.logger.error(f"Strategy Error: {e}")

            finally:
                self.queue.task_done()

# --- SYSTEM ENTRY POINT ---


async def main():
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

    # Initialize Channels
    event_queue = asyncio.Queue(maxsize=1000)  # Maxsize prevents memory overflow if consumer dies

    # Initialize Modules
    risk_gate = CircuitBreaker(max_latency_ms=50, max_drawdown_pct=0.02)
    order_manager = OrderManager()
    md_handler = MarketDataHandler("wss://api.exchange.com/stream", event_queue, ["BTC-USD"])
    strategy = MarketMakerStrategy(event_queue, risk_gate, order_manager)

    # Run the Reactor
    # gather() runs the producer and consumer concurrently
    await asyncio.gather(
        md_handler.connect(),
        strategy.run()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
