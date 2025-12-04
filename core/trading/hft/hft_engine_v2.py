"""
Module 1: High-Frequency Trading (HFT) Engine
=============================================

Architectural Blueprint:
------------------------
1.  **Concurrency**: Utilizes `asyncio` with `uvloop` (if available) for an event-driven, non-blocking architecture.
2.  **Algorithm**: Implements the Avellaneda-Stoikov market-making model using Reservation Price (r) and Optimal Spread (delta).
3.  **Resilience**: Implements a 3-state Circuit Breaker (Closed, Open, Half-Open) to manage system faults.
4.  **Networking**: Features a Zero-Copy Protocol design using `asyncio.Protocol` for minimizing latency.

Usage:
------
Run this module directly to simulate the trading loop.
"""

import asyncio
import random
import time
import math
import collections
from dataclasses import dataclass
from typing import List, Dict, Deque
from enum import Enum
import functools

# Try to import uvloop for performance
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False

# --- Configuration Constants ---
MAX_LATENCY_MS = 50
MAX_DRAWDOWN_PCT = 0.02
RISK_WINDOW_SECONDS = 300
MAX_INVENTORY = 100
DEFAULT_GAMMA = 0.1  # Risk aversion
DEFAULT_KAPPA = 1.5  # Liquidity parameter
CIRCUIT_FAILURE_THRESHOLD = 5
CIRCUIT_RECOVERY_TIMEOUT = 5.0  # seconds (short for simulation)

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

# --- 1.0 Zero-Copy Networking Protocol ---

class HFTRawProtocol(asyncio.Protocol):
    """
    Implements a zero-copy network protocol.
    Parses incoming bytes directly from the socket buffer.
    """
    def connection_made(self, transport):
        self.transport = transport
        # print("[Network] Connection established.")

    def data_received(self, data: bytes):
        # Zero-copy parsing using memoryview
        # In a real implementation, we would parse FIX or binary headers here.
        # For simulation, we just acknowledge receipt.
        mv = memoryview(data)
        # Example: Check first byte for message type
        # msg_type = mv[0]
        pass

    def connection_lost(self, exc):
        # print("[Network] Connection lost.")
        pass

# --- 3.0 System Resilience: Circuit Breaker ---

class CircuitBreakerOpenException(Exception):
    pass

class CircuitBreakerState(Enum):
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failure threshold reached, blocking calls
    HALF_OPEN = "HALF_OPEN" # Testing recovery

class CircuitBreaker:
    def __init__(self, failure_threshold: int, recovery_timeout: float):
        self.state = CircuitBreakerState.CLOSED
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.state != CircuitBreakerState.OPEN and self.failures >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            print(f"[CircuitBreaker] TRIPPED! State is now OPEN. Failures: {self.failures}")

    def record_success(self):
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.failures = 0
            print("[CircuitBreaker] RECOVERED. State is now CLOSED.")
        else:
            self.failures = 0

    def check_state(self):
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                print("[CircuitBreaker] Probing... State is now HALF_OPEN.")
            else:
                raise CircuitBreakerOpenException("Circuit is OPEN.")

# --- 2.0 Algorithmic Core: Avellaneda-Stoikov ---

class MarketMakerStrategy:
    """
    Implements the Avellaneda-Stoikov logic for inventory risk management.
    """
    def __init__(self, gamma: float = DEFAULT_GAMMA, kappa: float = DEFAULT_KAPPA):
        self.gamma = gamma  # Risk aversion
        self.kappa = kappa  # Liquidity parameter
        self.mid_prices: Deque[float] = collections.deque(maxlen=1000)

    def update_price(self, price: float):
        self.mid_prices.append(price)

    def calculate_volatility(self) -> float:
        if len(self.mid_prices) < 2:
            return 0.01 # Default fallback
        
        # Simple standard deviation of returns
        # In prod, optimize this to be incremental O(1)
        prices = list(self.mid_prices)
        mean = sum(prices) / len(prices)
        variance = sum((x - mean) ** 2 for x in prices) / len(prices)
        return math.sqrt(variance)

    def calculate_quotes(self, s: float, q: float, T_minus_t: float = 1.0) -> Dict[str, float]:
        """
        s: Mid-price
        q: Inventory (signed)
        T_minus_t: Time horizon (normalized to 1 for continuous markets)
        """
        sigma = self.calculate_volatility()
        sigma_sq = sigma ** 2

        # 1. Calculate Reservation Price (r)
        # r = s - q * gamma * sigma^2 * (T - t)
        reservation_price = s - (q * self.gamma * sigma_sq * T_minus_t)

        # 2. Calculate Optimal Spread (delta)
        # delta = (2 / gamma) * ln(1 + (gamma / kappa))
        # Note: We apply the volatility/time scaling often used in practice.
        # But keeping to the simplified formula from the blueprint text implies using gamma/kappa.
        
        # Spread calculation based on Avellaneda-Stoikov 2008 approx
        spread = (2 / self.gamma) * math.log(1 + (self.gamma / self.kappa))
        
        # Ensure minimum spread (tick size)
        spread = max(spread, 0.01)

        bid_price = reservation_price - (spread / 2)
        ask_price = reservation_price + (spread / 2)

        return {
            "r": reservation_price,
            "bid": bid_price,
            "ask": ask_price,
            "spread": spread,
            "sigma": sigma
        }

# --- Core Engine Components ---

class MarketDataHandler:
    def __init__(self, symbols: List[str], queue: asyncio.Queue, circuit: CircuitBreaker):
        self.symbols = symbols
        self.queue = queue
        self.circuit = circuit
        self.running = False

    async def _fetch_tick_sim(self):
        # 5% failure rate
        if random.random() < 0.05:
            raise TimeoutError("Network Timeout")
        
        # Simulate Network Latency
        await asyncio.sleep(random.uniform(0.001, 0.01))
        
        symbol = random.choice(self.symbols)
        base = 100.0
        # Add some trend and noise
        price = base + math.sin(time.time() * 0.1) * 10 + random.normalvariate(0, 0.5)
        return MarketTick(symbol, price - 0.05, price + 0.05, time.time())

    async def start(self):
        self.running = True
        print(f"[MarketDataHandler] Starting Feed (UVLOOP={UVLOOP_AVAILABLE})")
        while self.running:
            try:
                self.circuit.check_state()
                try:
                    tick = await self._fetch_tick_sim()
                    self.circuit.record_success()
                    await self.queue.put(tick)
                except TimeoutError:
                    self.circuit.record_failure()
            except CircuitBreakerOpenException:
                # Circuit is open, back off to avoid busy loop
                await asyncio.sleep(1.0)
            except Exception as e:
                print(f"[System] Error: {e}")
            
            # Reduce sleep to simulate HFT
            await asyncio.sleep(0.001)

    def stop(self):
        self.running = False

class OrderManager:
    def __init__(self, circuit: CircuitBreaker):
        self.orders: Dict[str, Order] = {}
        self.circuit = circuit
    
    async def place_order(self, order: Order):
        try:
            self.circuit.check_state()
            # Simulate Network IO
            await asyncio.sleep(0.002) 
            self.orders[order.id] = order
            self.circuit.record_success()
        except CircuitBreakerOpenException:
            # print(f"[OrderManager] REJECTED {order.id} - Circuit Open")
            pass
        except Exception:
            self.circuit.record_failure()

    async def cancel_all(self):
        # Emergency cancellation
        print("[OrderManager] Cancelling ALL orders...")
        self.orders.clear()

class HFTExecutionEngine:
    def __init__(self, symbol: str, start_balance: float):
        self.symbol = symbol
        self.balance = start_balance
        self.inventory = 0
        self.market_queue = asyncio.Queue()
        
        # Components
        self.circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=5.0)
        self.md_handler = MarketDataHandler([symbol], self.market_queue, self.circuit)
        self.order_manager = OrderManager(self.circuit)
        self.strategy = MarketMakerStrategy(gamma=0.5, kappa=2.0) # High gamma = hate inventory

    async def run(self):
        feed_task = asyncio.create_task(self.md_handler.start())
        print(f"[HFT] Engine Online. {self.symbol} | Bal: {self.balance}")
        
        tick_count = 0

        try:
            while True:
                # 1. Ingest Tick
                try:
                    tick = await asyncio.wait_for(self.market_queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    if self.circuit.state != CircuitBreakerState.OPEN:
                        print("[HFT] No data...")
                    continue

                tick_count += 1

                # 2. Strategy Update
                mid = (tick.bid + tick.ask) / 2
                self.strategy.update_price(mid)
                
                # 3. Calculate Quotes (Avellaneda-Stoikov)
                # Normalize inventory for the formula
                # T-t is constant 1.0 for perpetual operation
                quotes = self.strategy.calculate_quotes(mid, self.inventory, T_minus_t=1.0)
                
                bid = quotes['bid']
                ask = quotes['ask']
                r_price = quotes['r']
                
                # 4. Inventory Gate (Hard Constraint)
                if abs(self.inventory) >= MAX_INVENTORY:
                    if tick_count % 50 == 0:
                        print(f"[Risk] Max Inventory ({self.inventory}) Reached. Entering Reduce-Only.")
                    
                    # Logic to only place orders that reduce inventory
                    # We widen the spread on the entry side to discourage fills
                    if self.inventory > 0:
                        bid -= 1.0 # Lower bid to stop buying
                        ask -= 0.5 # Lower ask to encourage selling
                    else:
                        bid += 0.5
                        ask += 1.0
                
                # 5. Execution (Simulate)
                # Clear old orders (simplification for simulation)
                self.order_manager.orders.clear()
                
                await self.order_manager.place_order(Order(
                    id=f"B-{int(time.time()*1e6)}", symbol=self.symbol, side=OrderSide.BUY,
                    price=bid, quantity=1, timestamp=time.time()
                ))
                await self.order_manager.place_order(Order(
                    id=f"S-{int(time.time()*1e6)}", symbol=self.symbol, side=OrderSide.SELL,
                    price=ask, quantity=1, timestamp=time.time()
                ))

                # 6. Simulation of Fills
                # If market moves through our quotes
                if tick.ask < bid:
                    self.inventory += 1
                    self.balance -= bid
                    # print(f"-> FILLED BUY @ {bid:.2f} | Inv: {self.inventory}")
                elif tick.bid > ask:
                    self.inventory -= 1
                    self.balance += ask
                    # print(f"<- FILLED SELL @ {ask:.2f} | Inv: {self.inventory}")
                
                if tick_count % 100 == 0:
                     print(f"[Stats] Inv: {self.inventory} | Mid: {mid:.2f} | r: {r_price:.2f} | PnL (Unrealized): {self.balance + (self.inventory * mid) - 100000:.2f}")

        except KeyboardInterrupt:
            pass
        finally:
            self.md_handler.stop()
            print("Engine Shutdown.")

if __name__ == "__main__":
    engine = HFTExecutionEngine("BTC-USD", 100000.0)
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        pass
