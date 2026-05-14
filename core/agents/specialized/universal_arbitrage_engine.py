import math
from typing import Dict, Any, List, Tuple
from pydantic import BaseModel, Field

from core.agents.agent_base import AgentBase

class OrderBookLevel(BaseModel):
    price: float
    volume: float

class L2OrderBook(BaseModel):
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]

class ArbitrageResult(BaseModel):
    is_profitable: bool
    gross_spread: float
    net_profit: float
    ev: float
    ev_penalty: float
    volume_executed: float
    micro_price: float
    survival_probability: float
    execution_cost: float

class UniversalArbitrageEngine(AgentBase):
    """
    A specialized agent that parses L2 order book ladders to calculate exact slippage
    using book-walking algorithms. It computes a volume-weighted micro-price for fair value
    inference, subtracts granular execution costs, and utilizes an exponential sigmoid decay curve
    to estimate execution latency survival probability.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.maker_fee = config.get("maker_fee", 0.0001)
        self.taker_fee = config.get("taker_fee", 0.0002)
        self.latency_ms = config.get("latency_ms", 10.0)
        self.latency_decay_k = config.get("latency_decay_k", 0.1) # Steepness of decay
        self.latency_decay_midpoint = config.get("latency_decay_midpoint", 50.0) # Midpoint in ms

    def calculate_micro_price(self, order_book: L2OrderBook, depth: int = 5) -> float:
        """
        Computes volume-weighted micro-price using top levels of the order book
        for fair value inference based on order imbalance.
        """
        bids = order_book.bids[:depth]
        asks = order_book.asks[:depth]

        if not bids or not asks:
            return 0.0

        total_bid_vol = sum(b.volume for b in bids)
        total_ask_vol = sum(a.volume for a in asks)

        if total_bid_vol == 0 and total_ask_vol == 0:
            return 0.0

        # Volume-weighted average prices
        vwap_bids = sum(b.price * b.volume for b in bids) / total_bid_vol if total_bid_vol > 0 else 0
        vwap_asks = sum(a.price * a.volume for a in asks) / total_ask_vol if total_ask_vol > 0 else 0

        # Order imbalance based micro-price
        imbalance = total_bid_vol / (total_bid_vol + total_ask_vol)

        # Micro-price formula: I * VWAP_Ask + (1 - I) * VWAP_Bid
        # When bids are heavy (I->1), micro-price approaches Ask
        micro_price = (imbalance * vwap_asks) + ((1 - imbalance) * vwap_bids)

        return micro_price

    def calculate_survival_probability(self) -> float:
        """
        Utilizes an exponential sigmoid decay curve to estimate execution latency
        survival probability against adversarial AI and HFT sniper bots.
        P(survive) = 1 / (1 + e^(k * (latency - midpoint)))
        """
        exponent = self.latency_decay_k * (self.latency_ms - self.latency_decay_midpoint)
        # Cap exponent to prevent overflow
        exponent = max(min(exponent, 50), -50)
        survival_prob = 1.0 / (1.0 + math.exp(exponent))
        return survival_prob

    def walk_book(self, levels: List[OrderBookLevel], target_volume: float) -> Tuple[float, float, float]:
        """
        Book-walking algorithm to calculate exact slippage for a target volume.
        Returns (executed_volume, total_cost, average_price).
        """
        executed_volume = 0.0
        total_cost = 0.0

        for level in levels:
            remaining_volume = target_volume - executed_volume
            if remaining_volume <= 0:
                break

            vol_to_take = min(remaining_volume, level.volume)
            executed_volume += vol_to_take
            total_cost += vol_to_take * level.price

        avg_price = total_cost / executed_volume if executed_volume > 0 else 0.0
        return executed_volume, total_cost, avg_price

    def evaluate_arbitrage(self, exchange_a_book: L2OrderBook, exchange_b_book: L2OrderBook, target_volume: float) -> ArbitrageResult:
        """
        Evaluates profitability using gross volume rather than per-unit spreads.
        Requires processing full L2 order book ladders.
        """
        # We need a fair value reference point. We calculate micro-price on both and average.
        micro_price_a = self.calculate_micro_price(exchange_a_book)
        micro_price_b = self.calculate_micro_price(exchange_b_book)
        fair_value = (micro_price_a + micro_price_b) / 2.0

        # Determine the direction: Buy on A, Sell on B OR Buy on B, Sell on A
        # Let's check both directions.

        # Direction 1: Buy A, Sell B
        # To buy on A, we walk the asks of A. To sell on B, we walk the bids of B.
        exec_vol_buy_a, cost_a, avg_price_buy_a = self.walk_book(exchange_a_book.asks, target_volume)
        exec_vol_sell_b, revenue_b, avg_price_sell_b = self.walk_book(exchange_b_book.bids, target_volume)

        # Only execute the volume we can actually trade on both sides
        tradeable_vol_1 = min(exec_vol_buy_a, exec_vol_sell_b)

        # Recalculate exact costs for tradeable volume
        _, actual_cost_a, _ = self.walk_book(exchange_a_book.asks, tradeable_vol_1)
        _, actual_rev_b, _ = self.walk_book(exchange_b_book.bids, tradeable_vol_1)

        gross_profit_1 = actual_rev_b - actual_cost_a

        # Direction 2: Buy B, Sell A
        exec_vol_buy_b, cost_b, avg_price_buy_b = self.walk_book(exchange_b_book.asks, target_volume)
        exec_vol_sell_a, revenue_a, avg_price_sell_a = self.walk_book(exchange_a_book.bids, target_volume)

        tradeable_vol_2 = min(exec_vol_buy_b, exec_vol_sell_a)

        _, actual_cost_b, _ = self.walk_book(exchange_b_book.asks, tradeable_vol_2)
        _, actual_rev_a, _ = self.walk_book(exchange_a_book.bids, tradeable_vol_2)

        gross_profit_2 = actual_rev_a - actual_cost_b

        # Choose the better direction
        if tradeable_vol_1 > 0 and (gross_profit_1 > gross_profit_2 or tradeable_vol_2 == 0):
            chosen_vol = tradeable_vol_1
            gross_profit = gross_profit_1
            # We took liquidity on both sides (Taker/Taker)
            exec_cost = (actual_cost_a * self.taker_fee) + (actual_rev_b * self.taker_fee)
            spread = (actual_rev_b / chosen_vol) - (actual_cost_a / chosen_vol)
        elif tradeable_vol_2 > 0:
            chosen_vol = tradeable_vol_2
            gross_profit = gross_profit_2
            exec_cost = (actual_cost_b * self.taker_fee) + (actual_rev_a * self.taker_fee)
            spread = (actual_rev_a / chosen_vol) - (actual_cost_b / chosen_vol)
        else:
            return ArbitrageResult(
                is_profitable=False, gross_spread=0.0, net_profit=0.0,
                ev=0.0, ev_penalty=0.0, volume_executed=0.0,
                micro_price=fair_value, survival_probability=0.0, execution_cost=0.0
            )

        net_profit = gross_profit - exec_cost
        survival_prob = self.calculate_survival_probability()

        # EV = P(survive) * NetProfit - (1 - P(survive)) * Loss
        # If we fail, we might assume we eat the execution cost as a penalty, or the spread moves against us.
        # Let's say EV penalty is simply the loss of the trade + fees if we get front-run.
        # A simple model: if we fail, we take a loss equal to the gross spread.
        failure_penalty = abs(gross_profit) + exec_cost
        ev = (survival_prob * net_profit) - ((1 - survival_prob) * failure_penalty)
        ev_penalty = (1 - survival_prob) * failure_penalty

        is_profitable = ev > 0

        return ArbitrageResult(
            is_profitable=is_profitable,
            gross_spread=spread,
            net_profit=net_profit,
            ev=ev,
            ev_penalty=ev_penalty,
            volume_executed=chosen_vol,
            micro_price=fair_value,
            survival_probability=survival_prob,
            execution_cost=exec_cost
        )

    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        exchange_a_data = kwargs.get("exchange_a_book")
        exchange_b_data = kwargs.get("exchange_b_book")
        target_volume = kwargs.get("target_volume", 1.0)

        if not exchange_a_data or not exchange_b_data:
            return {"error": "Missing L2 order book data"}

        book_a = L2OrderBook(**exchange_a_data)
        book_b = L2OrderBook(**exchange_b_data)

        result = self.evaluate_arbitrage(book_a, book_b, target_volume)

        return {
            "status": "success",
            "result": result.model_dump()
        }
