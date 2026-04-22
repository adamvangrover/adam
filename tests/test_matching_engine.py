import unittest
from uuid import uuid4
from datetime import datetime
from core.quantitative.matching_engine import MatchingEngine, OrderBook, RUST_AVAILABLE
from core.unified_ledger.schema import ChildOrder, OrderSide, OrderType, ExecutionVenue

class TestMatchingEngine(unittest.TestCase):
    def setUp(self):
        self.engine = MatchingEngine()

    def test_initialization(self):
        self.assertIsNotNone(self.engine)
        if RUST_AVAILABLE:
            self.assertTrue(self.engine.use_rust)
        else:
            self.assertFalse(self.engine.use_rust)

    def test_process_order(self):
        order1 = ChildOrder(
            parent_id=uuid4(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            order_type=OrderType.LIMIT,
            venue=ExecutionVenue.INTERNAL,
            desk_id="DESK1"
        )

        result1 = self.engine.process_order(order1)
        self.assertEqual(result1["status"], "PENDING")
        self.assertEqual(result1["filled_quantity"], 0.0)

        order2 = ChildOrder(
            parent_id=uuid4(),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=50,
            price=149.0,
            order_type=OrderType.LIMIT,
            venue=ExecutionVenue.INTERNAL,
            desk_id="DESK2"
        )

        result2 = self.engine.process_order(order2)
        self.assertEqual(result2["status"], "FILLED")
        self.assertEqual(result2["filled_quantity"], 50.0)
        self.assertEqual(len(result2["fills"]), 1)

        # In this specific test, order1 (BUY 150) should be resting.
        # order2 (SELL 149) should match against order1 at 150.
        self.assertEqual(result2["fills"][0]["price"], 150.0)

    def test_cancel_order(self):
        order = ChildOrder(
            parent_id=uuid4(),
            symbol="MSFT",
            side=OrderSide.BUY,
            quantity=100,
            price=300.0,
            order_type=OrderType.LIMIT,
            venue=ExecutionVenue.INTERNAL,
            desk_id="DESK1"
        )

        self.engine.process_order(order)

        # Test cancellation
        success = self.engine.cancel_order("MSFT", order.order_id)
        self.assertTrue(success)

        # Try to match against cancelled order
        order2 = ChildOrder(
            parent_id=uuid4(),
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=50,
            price=299.0,
            order_type=OrderType.LIMIT,
            venue=ExecutionVenue.INTERNAL,
            desk_id="DESK2"
        )

        result2 = self.engine.process_order(order2)
        self.assertEqual(result2["status"], "PENDING")
        self.assertEqual(result2["filled_quantity"], 0.0)

if __name__ == '__main__':
    unittest.main()
