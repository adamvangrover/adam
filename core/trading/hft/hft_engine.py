"""
Module 1: High-Frequency Trading (HFT) Engine
=============================================

Architect Notes:
----------------
1.  **Concurrency Model**: We use `asyncio` because HFT at this level (market making) is network-bound, not CPU-bound.
    Waiting for WebSocket updates or REST API confirmations wastes cycles. `asyncio` allows us to handle thousands
    of concurrent connections on a single thread without the context-switching overhead of OS threads.
2.  **State Management**: The `OrderManager` uses a dictionary for O(1) lookups. In a production C++ system,
    we would use a ring buffer or similar lock-free structure, but for Python, a dict is sufficient for prototyping.
3.  **Risk Controls**: The `CircuitBreaker` is hard-coded and checked *before* every order placement.
    It tracks drawdown and latency. This is "Risk-First" architecture.

Usage:
------
Run this module directly to simulate the trading loop.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

# --- Constants & Configuration ---
MAX_LATENCY_MS = 50
MAX_DRAWDOWN_PCT = 0.02
RISK_WINDOW_SECONDS = 300  # 5 minutes

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"

@dataclass
class Order:
    id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    timestamp: float
    status: OrderStatus = OrderStatus.OPEN

@dataclass
class MarketTick:
    symbol: str
    bid: float
    ask: float
    timestamp: float

# --- Components ---

class CircuitBreaker:
    """
    Critical Risk Component.
    Monitors system health and PnL. Halts trading if thresholds are breached.
    """
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.is_tripped = False
        self.trip_reason = ""

    def update_pnl(self, current_balance: float):
        self.current_balance = current_balance
        self.peak_balance = max(self.peak_balance, self.current_balance)

        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if drawdown > MAX_DRAWDOWN_PCT:
            self.is_tripped = True
            self.trip_reason = f"Drawdown limit exceeded: {drawdown:.2%}"

    def check_latency(self, latency_ms: float):
        if latency_ms > MAX_LATENCY_MS:
            self.is_tripped = True
            self.trip_reason = f"Latency limit exceeded: {latency_ms}ms"

    def can_trade(self) -> bool:
        return not self.is_tripped

class MarketDataHandler:
    """
    Simulates WebSocket ingestion.
    In production, this would wrap `websockets` or `aiohttp` to connect to an exchange.
    """
    def __init__(self, symbols: List[str], queue: asyncio.Queue):
        self.symbols = symbols
        self.queue = queue
        self.running = False

    async def start(self):
        self.running = True
        print("[MarketDataHandler] Starting WebSocket feed simulation...")
        while self.running:
            # Simulate network latency and tick arrival
            await asyncio.sleep(random.uniform(0.001, 0.1))

            # Generate synthetic tick
            symbol = random.choice(self.symbols)
            mid_price = 100.0 + random.uniform(-1, 1) # Random walk around 100
            spread = 0.05

            tick = MarketTick(
                symbol=symbol,
                bid=mid_price - (spread / 2),
                ask=mid_price + (spread / 2),
                timestamp=time.time()
            )

            await self.queue.put(tick)

    def stop(self):
        self.running = False

class OrderManager:
    """
    Manages order state.
    """
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.fills: List[Order] = []

    def place_order(self, order: Order):
        print(f"[OrderManager] Placing {order.side.value} {order.quantity} @ {order.price:.2f}")
        self.orders[order.id] = order
        # In a real system, we'd send to exchange API here

    def simulate_fill(self, order_id: str):
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = OrderStatus.FILLED
            self.fills.append(order)
            print(f"[OrderManager] Order {order_id} FILLED")
            del self.orders[order_id]

    def cancel_order(self, order_id: str):
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            del self.orders[order_id]

# --- Main Strategy Engine ---

class HFTStrategy:
    def __init__(self, symbol: str, balance: float):
        self.symbol = symbol
        self.market_queue = asyncio.Queue()
        self.md_handler = MarketDataHandler([symbol], self.market_queue)
        self.order_manager = OrderManager()
        self.risk_gate = CircuitBreaker(balance)
        self.position = 0
        self.cash = balance

    async def run(self):
        # Start data feed in background
        feed_task = asyncio.create_task(self.md_handler.start())

        print(f"[HFTStrategy] Engine initialized for {self.symbol}. Balance: ${self.cash}")

        try:
            while True:
                # 1. Check Risk Gate
                if not self.risk_gate.can_trade():
                    print(f"[RISK ALERT] Circuit Breaker Tripped! Reason: {self.risk_gate.trip_reason}")
                    print("Halting all trading.")
                    self.md_handler.stop()
                    break

                # 2. Process Market Data
                try:
                    # Timeout to simulate periodic housekeeping if no ticks
                    tick = await asyncio.wait_for(self.market_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Measure processing latency (simulated)
                start_time = time.time()
                latency_ms = (start_time - tick.timestamp) * 1000

                # Check latency risk
                self.risk_gate.check_latency(latency_ms)

                # 3. Strategy Logic (Market Making)
                # Ladder limit orders around the mid-price
                spread = tick.ask - tick.bid
                mid_price = (tick.ask + tick.bid) / 2

                # Simple logic: If we have no open orders, place a bracket
                if not self.order_manager.orders:
                    # Buy below bid
                    buy_price = tick.bid - (spread * 0.1)
                    # Sell above ask
                    sell_price = tick.ask + (spread * 0.1)

                    buy_order = Order(
                        id=f"B-{int(time.time()*1000)}",
                        symbol=self.symbol,
                        side=OrderSide.BUY,
                        price=buy_price,
                        quantity=10,
                        timestamp=time.time()
                    )

                    sell_order = Order(
                        id=f"S-{int(time.time()*1000)}",
                        symbol=self.symbol,
                        side=OrderSide.SELL,
                        price=sell_price,
                        quantity=10,
                        timestamp=time.time()
                    )

                    self.order_manager.place_order(buy_order)
                    self.order_manager.place_order(sell_order)

                # 4. Simulate Fills (Mock Exchange Matching)
                # In real life, this comes from the WebSocket feed as "ExecutionReport"
                # Here we simulate: if price moves through our limit, we fill.
                to_fill = []
                for oid, order in self.order_manager.orders.items():
                    if order.side == OrderSide.BUY and tick.ask <= order.price:
                        to_fill.append(oid)
                    elif order.side == OrderSide.SELL and tick.bid >= order.price:
                        to_fill.append(oid)

                for oid in to_fill:
                    self.order_manager.simulate_fill(oid)
                    # Update PnL (Mock)
                    # Assume we capture the spread on every round trip
                    self.cash += random.uniform(-5, 15) # Random PnL swing
                    self.risk_gate.update_pnl(self.cash)

                # Simulate processing time
                await asyncio.sleep(0.005)

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            feed_task.cancel()
            print("HFT Engine Shutdown.")

if __name__ == "__main__":
    # Entry point
    engine = HFTStrategy(symbol="BTCUSD", balance=100000.0)
    asyncio.run(engine.run())
