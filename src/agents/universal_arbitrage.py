import math
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel


class OrderBookLevel(BaseModel):
    price: float
    volume: float


class OrderBook(BaseModel):
    exchange: str
    symbol: str
    bids: List[OrderBookLevel]  # Ordered best to worst (descending)
    asks: List[OrderBookLevel]  # Ordered best to worst (ascending)
    taker_fee_bps: float
    maker_fee_bps: float
    network_latency_ms: float


class ArbitrageExecutionPlan(BaseModel):
    symbol: str
    buy_exchange: str
    sell_exchange: str
    target_volume: float
    fair_value_micro_price: float
    gross_revenue: float
    net_profit: float
    latency_survival_probability: float
    expected_value: float
    is_executable: bool


class UniversalArbitrageEngine:
    """
    Advanced institutional arbitrage engine operating on L2 order book data,
    incorporating book walking, fair value micro-price, and adversarial latency decay.
    """

    def __init__(
        self, min_ev: float = 0.0, adversarial_latency_halflife_ms: float = 5.0
    ):
        self.min_ev = min_ev
        # The time it takes for our probability of executing a profitable trade to drop by 50%
        # assuming a perfectly adversarial environment where other AIs are competing.
        self.adversarial_latency_halflife_ms = adversarial_latency_halflife_ms

    def _walk_book(
        self, levels: List[OrderBookLevel], target_volume: float
    ) -> Tuple[float, float]:
        """
        Walks the order book to fill the target volume.
        Returns (executed_volume, total_cost_or_revenue)
        """
        executed_vol = 0.0
        total_value = 0.0

        for level in levels:
            rem_vol = target_volume - executed_vol
            if rem_vol <= 0:
                break

            fill_vol = min(level.volume, rem_vol)
            executed_vol += fill_vol
            total_value += fill_vol * level.price

        return executed_vol, total_value

    def _calculate_micro_price(self, book: OrderBook) -> float:
        """
        Calculates the volume-weighted micro-price (Fair Value) based on the best bid/ask imbalance.
        """
        if not book.bids or not book.asks:
            return 0.0

        best_bid = book.bids[0]
        best_ask = book.asks[0]

        imbalance = best_bid.volume / (best_bid.volume + best_ask.volume)
        # Micro-price interpolates between bid and ask based on imbalance.
        # High bid volume = price closer to ask.
        micro_price = (best_ask.price * imbalance) + (best_bid.price * (1 - imbalance))
        return micro_price

    def _calculate_adversarial_survival(self, latency_ms: float) -> float:
        """
        Models the probability that our trade executes before an adversarial AI snipes it.
        Uses exponential decay. P(Survival) = 0.5 ^ (latency / halflife)
        """
        return math.pow(0.5, latency_ms / self.adversarial_latency_halflife_ms)

    def analyze_market(
        self, order_books: List[OrderBook], trade_sizes: Optional[List[float]] = None
    ) -> List[ArbitrageExecutionPlan]:
        if trade_sizes is None:
            trade_sizes = [0.1, 1.0, 5.0, 10.0]

        opportunities = []

        # Group by symbol
        grouped_books: Dict[str, List[OrderBook]] = {}
        for book in order_books:
            grouped_books.setdefault(book.symbol, []).append(book)

        for symbol, books in grouped_books.items():
            if len(books) < 2:
                continue

            # Compare every pair of exchanges
            for i in range(len(books)):
                for j in range(len(books)):
                    if i == j:
                        continue

                    buy_book = books[i]
                    sell_book = books[j]

                    # For different possible trade sizes
                    for target_vol in trade_sizes:
                        # 1. Walk the ASK book to BUY
                        buy_vol, buy_cost = self._walk_book(buy_book.asks, target_vol)

                        # 2. Walk the BID book to SELL
                        sell_vol, sell_rev = self._walk_book(sell_book.bids, target_vol)

                        # We can only execute the minimum available volume across both books
                        executable_vol = min(buy_vol, sell_vol)
                        if executable_vol <= 0 or executable_vol < target_vol:
                            continue  # Book too thin for this size

                        # Recalculate exactly for the executable volume
                        _, exact_buy_cost = self._walk_book(
                            buy_book.asks, executable_vol
                        )
                        _, exact_sell_rev = self._walk_book(
                            sell_book.bids, executable_vol
                        )

                        gross_revenue = exact_sell_rev - exact_buy_cost

                        # 3. Factor in Execution Costs (Taker fees since we cross the spread)
                        buy_fee = exact_buy_cost * (buy_book.taker_fee_bps / 10000.0)
                        sell_fee = exact_sell_rev * (sell_book.taker_fee_bps / 10000.0)

                        net_profit = gross_revenue - buy_fee - sell_fee

                        # 4. Fair Value (Average of micro-prices of both venues)
                        mp_buy = self._calculate_micro_price(buy_book)
                        mp_sell = self._calculate_micro_price(sell_book)
                        fv = (mp_buy + mp_sell) / 2.0

                        # 5. Adversarial Risk Decay
                        # Total latency is round trip to both venues
                        total_latency = (
                            buy_book.network_latency_ms + sell_book.network_latency_ms
                        )
                        prob_success = self._calculate_adversarial_survival(
                            total_latency
                        )

                        # Expected Value (EV) = (Profit * P(Success)) - (Cost_of_Failed_Leg * P(Failure))
                        # For simplicity, assuming if we fail we lose the taker fees of a broken leg as penalty
                        penalty = buy_fee + sell_fee
                        ev = (net_profit * prob_success) - (
                            penalty * (1 - prob_success)
                        )

                        if ev > self.min_ev:
                            plan = ArbitrageExecutionPlan(
                                symbol=symbol,
                                buy_exchange=buy_book.exchange,
                                sell_exchange=sell_book.exchange,
                                target_volume=executable_vol,
                                fair_value_micro_price=round(fv, 4),
                                gross_revenue=round(gross_revenue, 4),
                                net_profit=round(net_profit, 4),
                                latency_survival_probability=round(prob_success, 4),
                                expected_value=round(ev, 4),
                                is_executable=True,
                            )
                            opportunities.append(plan)

        return sorted(opportunities, key=lambda x: x.expected_value, reverse=True)

    def execute(self, **kwargs):
        books_data = kwargs.get("order_books", [])
        trade_sizes = kwargs.get("trade_sizes", [0.1, 1.0, 5.0, 10.0])
        books = [OrderBook(**b) if isinstance(b, dict) else b for b in books_data]
        return self.analyze_market(books, trade_sizes)
