"""
Avellaneda-Stoikov Market Making Engine
=======================================

This module implements the stochastic control framework for market making as described in the
Unified Financial Operating System blueprint.

Theoretical Foundation:
-----------------------
- Reservation Price (r): The price where the MM is indifferent between buying and selling.
  r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

- Optimal Spread (delta): Half-spread distance from reservation price.
  delta(s, q, t) = (1 / gamma) * ln(1 + gamma / kappa) + 0.5 * gamma * sigma^2 * (T - t)

Parameters:
-----------
- s: Mid price
- q: Inventory (signed)
- t: Current time
- T: Terminal time (end of trading session)
- gamma: Risk aversion parameter
- sigma: Volatility
- kappa: Order book liquidity density

"""

import asyncio
import math
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import numpy as np

# Reuse basic structures from hft_engine
from core.trading.hft.hft_engine import (
    Order, OrderSide, OrderStatus, MarketTick, CircuitBreaker,
    MarketDataHandler, OrderManager
)

class AvellanedaStoikovStrategy:
    def __init__(
        self,
        symbol: str,
        balance: float,
        gamma: float = 0.1,
        kappa: float = 1.5,
        sigma: float = 0.05,
        session_duration: float = 3600  # 1 hour session for T
    ):
        self.symbol = symbol
        self.market_queue = asyncio.Queue()
        self.md_handler = MarketDataHandler([symbol], self.market_queue)
        self.order_manager = OrderManager()
        self.risk_gate = CircuitBreaker(balance)

        # State
        self.position = 0
        self.cash = balance
        self.start_time = time.time()
        self.end_time = self.start_time + session_duration

        # Model Parameters
        self.gamma = gamma  # Risk aversion
        self.kappa = kappa  # Liquidity parameter
        self.sigma = sigma  # Volatility (should be dynamic in prod)

        # History for Volatility Estimation
        self.price_history = []

    def estimate_volatility(self, new_price: float, window: int = 100):
        """
        Updates sigma based on rolling standard deviation of returns.
        """
        self.price_history.append(new_price)
        if len(self.price_history) > window:
            self.price_history.pop(0)

        if len(self.price_history) > 10:
            returns = np.diff(np.log(self.price_history))
            self.sigma = np.std(returns) * math.sqrt(252 * 24 * 3600) # Annualized?
            # For HFT, we usually scale to the tick frequency.
            # Let's keep it simple: raw std dev of recent ticks for now, scaled up slightly.
            self.sigma = np.std(self.price_history) # Simplified proxy

    def calculate_quotes(self, s: float, t: float):
        """
        Calculates optimal bid and ask prices.
        """
        T = self.end_time
        q = self.position

        time_left = max(0, T - t)

        # Avoid division by zero or log errors
        if self.gamma <= 0: self.gamma = 0.01
        if self.kappa <= 0: self.kappa = 0.1

        # 1. Reservation Price
        # r = s - q * gamma * sigma^2 * (T - t)
        reservation_price = s - (q * self.gamma * (self.sigma ** 2) * time_left)

        # 2. Optimal Spread (Half-Spread)
        # delta = (1/gamma) * ln(1 + gamma/kappa) + 0.5 * gamma * sigma^2 * (T - t)
        spread_term_1 = (1 / self.gamma) * math.log(1 + (self.gamma / self.kappa))
        spread_term_2 = 0.5 * self.gamma * (self.sigma ** 2) * time_left
        half_spread = spread_term_1 + spread_term_2

        optimal_bid = reservation_price - half_spread
        optimal_ask = reservation_price + half_spread

        return optimal_bid, optimal_ask

    async def run(self):
        feed_task = asyncio.create_task(self.md_handler.start())
        print(f"[AS-Engine] Started for {self.symbol}. Gamma={self.gamma}, Kappa={self.kappa}")

        try:
            while True:
                current_time = time.time()

                # Check session end
                if current_time >= self.end_time:
                    print("Session ended. Flattening inventory.")
                    # In real logic: dump inventory.
                    break

                # 1. Check Risk
                if not self.risk_gate.can_trade():
                    print(f"RISK HALT: {self.risk_gate.trip_reason}")
                    break

                # 2. Get Market Tick
                try:
                    tick = await asyncio.wait_for(self.market_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                mid_price = (tick.bid + tick.ask) / 2.0
                self.estimate_volatility(mid_price)

                # 3. Calculate AS Quotes
                bid_price, ask_price = self.calculate_quotes(mid_price, current_time)

                # Round to 2 decimals
                bid_price = round(bid_price, 2)
                ask_price = round(ask_price, 2)

                # 4. Update Orders
                # In this simplified engine, we just cancel all and replace.
                # In prod, we would diff and modify.
                to_cancel = list(self.order_manager.orders.keys())
                for oid in to_cancel:
                    self.order_manager.cancel_order(oid)

                # Place new orders
                # Quote Filtering: Don't cross the market (unless taking liquidity, but this is a maker strategy)
                if bid_price >= tick.ask: bid_price = tick.ask - 0.01
                if ask_price <= tick.bid: ask_price = tick.bid + 0.01

                # Don't quote negative spreads
                if bid_price >= ask_price:
                    ask_price = bid_price + 0.02

                buy_order = Order(
                    id=f"B-{int(time.time()*10000)}",
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    price=bid_price,
                    quantity=10,
                    timestamp=time.time()
                )

                sell_order = Order(
                    id=f"S-{int(time.time()*10000)}",
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    price=ask_price,
                    quantity=10,
                    timestamp=time.time()
                )

                self.order_manager.place_order(buy_order)
                self.order_manager.place_order(sell_order)

                # 5. Simulate Matching (Mock)
                # If market moves through our quotes
                # Current simulated tick spread is tight, so we might get filled.

                filled_ids = []
                for oid, order in self.order_manager.orders.items():
                    if order.side == OrderSide.BUY and tick.ask <= order.price:
                        # Filled on Buy
                        self.position += order.quantity
                        self.cash -= (order.price * order.quantity)
                        filled_ids.append(oid)
                        print(f"FILLED BUY @ {order.price}. Inventory: {self.position}")

                    elif order.side == OrderSide.SELL and tick.bid >= order.price:
                        # Filled on Sell
                        self.position -= order.quantity
                        self.cash += (order.price * order.quantity)
                        filled_ids.append(oid)
                        print(f"FILLED SELL @ {order.price}. Inventory: {self.position}")

                for oid in filled_ids:
                    self.order_manager.simulate_fill(oid)

                # Update Risk Gate
                mark_to_market = self.cash + (self.position * mid_price)
                self.risk_gate.update_pnl(mark_to_market)

                await asyncio.sleep(0.01)

        except KeyboardInterrupt:
            pass
        finally:
            feed_task.cancel()

if __name__ == "__main__":
    strategy = AvellanedaStoikovStrategy(symbol="AAPL", balance=1000000.0)
    asyncio.run(strategy.run())
