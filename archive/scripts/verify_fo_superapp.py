import unittest
import asyncio
import os
from core.engine.meta_orchestrator import MetaOrchestrator
from core.memory.engine import MemoryEngine

from unittest.mock import MagicMock


class TestFOSuperAppIntegration(unittest.TestCase):

    def setUp(self):
        mock_legacy = MagicMock()
        self.orchestrator = MetaOrchestrator(legacy_orchestrator=mock_legacy)

    def test_market_routing(self):
        """Test that 'Price MSFT' routes to FO Market Module."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            self.orchestrator.route_request("Get price for MSFT")
        )
        self.assertEqual(result["status"], "FO Market Data Retrieved")
        self.assertEqual(result["data"]["symbol"], "MSFT")
        self.assertTrue("price" in result["data"])
        loop.close()

    def test_execution_routing(self):
        """Test that 'Buy 500 AAPL' routes to FO Execution Module."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            self.orchestrator.route_request("Buy 500 AAPL")
        )
        self.assertEqual(result["status"], "FO Execution Submitted")
        self.assertEqual(result["report"]["filled_qty"], 500.0)
        loop.close()

    def test_memory_persistence(self):
        """Test that MemoryEngine persists data to a file."""
        db_path = "tests/test_memory.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        mem1 = MemoryEngine(db_path=db_path)
        mem1.store_memory("Persistent memory check", "test")

        mem2 = MemoryEngine(db_path=db_path)
        results = mem2.query_memory("Persistent")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "Persistent memory check")

        if os.path.exists(db_path):
            os.remove(db_path)

    def test_ips_generation(self):
        """Test routing for IPS/Governance."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            self.orchestrator.route_request("Generate IPS for the Smith Family")
        )
        self.assertEqual(result["status"], "IPS Generated")
        self.assertIn("Investment Policy Statement", result["ips"])
        loop.close()

    def test_deal_screening(self):
        """Test routing for Deal Screening."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # "Screen deal 100 10" (val 100, ebitda 10 -> 10x)
        result = loop.run_until_complete(
            self.orchestrator.route_request("Screen deal ProjectX sector Tech val 100 ebitda 10")
        )
        self.assertEqual(result["status"], "Deal Screened")
        self.assertEqual(result["result"]["implied_multiple"], 10.0)
        self.assertIn("ai_due_diligence", result["result"])
        loop.close()

    def test_wealth_simulation(self):
        """Direct unit test of WealthManager MC simulation."""
        from core.family_office.wealth_manager import WealthManager
        wm = WealthManager()
        plan = wm.plan_goal("Test", 100000, 10, 50000)
        self.assertIn("probability_of_success", plan)
        self.assertTrue(0 <= plan["probability_of_success"] <= 1)

    def test_portfolio_risk(self):
        """Direct unit test of PortfolioAggregator risk integration."""
        from core.family_office.portfolio import PortfolioAggregator
        pa = PortfolioAggregator()
        res = pa.aggregate_risk([{"name": "Fund A", "aum": 1000000}])
        self.assertIn("stress_tests", res)
        self.assertIn("daily_var_95", res)


if __name__ == '__main__':
    unittest.main()
