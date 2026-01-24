import heapq
from collections import deque
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from uuid import UUID
import logging

# Adjust import based on where this file is relative to repo root
from core.unified_ledger.schema import ChildOrder, OrderSide, OrderType, ExecutionVenue

logger = logging.getLogger(__name__)

class OrderBook:
    def __init__(self, symbol: str):
        self.symbol = symbol
        # Bids: Max-Heap (stores negative price because heapq is min-heap)
        self.bids_heap = []
        # Asks: Min-Heap
        self.asks_heap = []

        # Maps price -> Deque of orders (Time Priority)
        self.bids_levels: Dict[float, deque] = {}
        self.asks_levels: Dict[float, deque] = {}

        # Map order_id -> ChildOrder (for cancellation)
        self.order_map = {}

    def cancel_order(self, order_id: UUID) -> bool:
        """
        Cancels an order in O(1) by removing from map and marking status.
        The order remains in the heap/deque until it reaches the top (Lazy Deletion).
        """
        if order_id in self.order_map:
            order = self.order_map[order_id]
            order.status = "CANCELED"
            del self.order_map[order_id]
            return True
        return False

    def add_order(self, order: ChildOrder) -> List[Tuple[ChildOrder, float]]:
        """
        Matches incoming order against book. Returns list of fills (matched_order, fill_qty).
        If not fully filled and LIMIT, adds remainder to book.
        """
        fills = []
        remaining_qty = order.quantity

        if order.side == OrderSide.BUY:
            # Match against Asks
            while remaining_qty > 0 and self.asks_heap:
                best_ask_price = self.asks_heap[0]

                # Check Limit Price
                if order.order_type == OrderType.LIMIT and order.price is not None:
                    if best_ask_price > order.price:
                        break # Best ask is too expensive

                # Get the queue for this price
                level_queue = self.asks_levels[best_ask_price]

                if not level_queue:
                    # Cleanup empty level if heap sync failed
                    heapq.heappop(self.asks_heap)
                    continue

                best_ask_order = level_queue[0]

                # Lazy Deletion Check
                if best_ask_order.status == "CANCELED":
                    level_queue.popleft()
                    # Map is already cleaned in cancel_order
                    if not level_queue:
                        del self.asks_levels[best_ask_price]
                        heapq.heappop(self.asks_heap)
                    continue

                # Calculate Match Quantity
                match_qty = min(remaining_qty, best_ask_order.quantity - best_ask_order.filled_quantity)

                # Execute Match
                fills.append((best_ask_order, match_qty))

                # Update Best Ask Order
                best_ask_order.filled_quantity += match_qty

                # Update Incoming Order Tracking
                order.filled_quantity += match_qty
                remaining_qty -= match_qty

                # Remove if filled
                if best_ask_order.filled_quantity >= best_ask_order.quantity:
                    level_queue.popleft()
                    # Clean up map
                    if best_ask_order.order_id in self.order_map:
                        del self.order_map[best_ask_order.order_id]

                    if not level_queue:
                        del self.asks_levels[best_ask_price]
                        heapq.heappop(self.asks_heap)

            # If remainder exists and is Limit, add to book
            if remaining_qty > 0 and order.order_type == OrderType.LIMIT:
                self._add_bid(order)

        elif order.side == OrderSide.SELL:
            # Match against Bids
            while remaining_qty > 0 and self.bids_heap:
                best_bid_price = -self.bids_heap[0] # Flip sign back

                if order.order_type == OrderType.LIMIT and order.price is not None:
                    if best_bid_price < order.price:
                        break # Best bid is too low

                level_queue = self.bids_levels[best_bid_price]

                if not level_queue:
                    heapq.heappop(self.bids_heap)
                    continue

                best_bid_order = level_queue[0]

                # Lazy Deletion Check
                if best_bid_order.status == "CANCELED":
                    level_queue.popleft()
                    if not level_queue:
                        del self.bids_levels[best_bid_price]
                        heapq.heappop(self.bids_heap)
                    continue

                match_qty = min(remaining_qty, best_bid_order.quantity - best_bid_order.filled_quantity)

                fills.append((best_bid_order, match_qty))

                best_bid_order.filled_quantity += match_qty
                order.filled_quantity += match_qty
                remaining_qty -= match_qty

                if best_bid_order.filled_quantity >= best_bid_order.quantity:
                    level_queue.popleft()
                    if best_bid_order.order_id in self.order_map:
                        del self.order_map[best_bid_order.order_id]

                    if not level_queue:
                        del self.bids_levels[best_bid_price]
                        heapq.heappop(self.bids_heap)

            if remaining_qty > 0 and order.order_type == OrderType.LIMIT:
                self._add_ask(order)

        return fills

    def _add_bid(self, order: ChildOrder):
        price = order.price
        if price not in self.bids_levels:
            self.bids_levels[price] = deque()
            heapq.heappush(self.bids_heap, -price) # Max heap
        self.bids_levels[price].append(order)
        self.order_map[order.order_id] = order

    def _add_ask(self, order: ChildOrder):
        price = order.price
        if price not in self.asks_levels:
            self.asks_levels[price] = deque()
            heapq.heappush(self.asks_heap, price) # Min heap
        self.asks_levels[price].append(order)
        self.order_map[order.order_id] = order

    def get_l2_snapshot(self, depth: int = 5):
        # NOTE: This snapshot might include CANCELED orders if they haven't been lazily removed.
        # For a perfect snapshot, one would need to filter them, but O(N) cost.
        # We accept this for simulation speed, or we can filter in the loop.
        bids = []
        temp_bids = self.bids_heap[:]
        count = 0
        while temp_bids and count < depth:
            p = -heapq.heappop(temp_bids)
            # Sum only non-canceled
            valid_orders = [o for o in self.bids_levels[p] if o.status != "CANCELED"]
            if not valid_orders:
                continue
            qty = sum(o.quantity - o.filled_quantity for o in valid_orders)
            bids.append((p, qty))
            count += 1

        asks = []
        temp_asks = self.asks_heap[:]
        count = 0
        while temp_asks and count < depth:
            p = heapq.heappop(temp_asks)
            valid_orders = [o for o in self.asks_levels[p] if o.status != "CANCELED"]
            if not valid_orders:
                continue
            qty = sum(o.quantity - o.filled_quantity for o in valid_orders)
            asks.append((p, qty))
            count += 1

        return {"symbol": self.symbol, "bids": bids, "asks": asks}

class MatchingEngine:
    """
    Manages order books for multiple symbols.
    """
    def __init__(self):
        self.books: Dict[str, OrderBook] = {}

    def cancel_order(self, symbol: str, order_id: UUID) -> bool:
        if symbol in self.books:
            return self.books[symbol].cancel_order(order_id)
        return False

    def process_order(self, order: ChildOrder) -> Dict:
        if order.symbol not in self.books:
            self.books[order.symbol] = OrderBook(order.symbol)

        book = self.books[order.symbol]
        fills = book.add_order(order)

        # Calculate average fill price
        total_fill_cost = 0.0
        total_fill_qty = 0.0

        processed_fills = []

        for matched_order, qty in fills:
            price = matched_order.price # Limit price of maker
            cost = price * qty
            total_fill_cost += cost
            total_fill_qty += qty
            processed_fills.append({
                "maker_order_id": str(matched_order.order_id),
                "taker_order_id": str(order.order_id),
                "price": price,
                "quantity": qty,
                "timestamp": datetime.utcnow().isoformat()
            })

        if total_fill_qty > 0:
            avg_price = total_fill_cost / total_fill_qty
            order.average_fill_price = avg_price
            order.status = "FILLED" if order.filled_quantity >= order.quantity else "PARTIALLY_FILLED"

        return {
            "order_id": str(order.order_id),
            "symbol": order.symbol,
            "status": order.status,
            "filled_quantity": order.filled_quantity,
            "fills": processed_fills
        }
