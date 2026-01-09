import unittest
import os
import json
import asyncio
from core.agents.specialized.market_update_agent import MarketUpdateAgent
from core.market_data.manager import MarketDataManager

class TestDynamicUpdates(unittest.TestCase):

    def test_agent_update(self):
        agent = MarketUpdateAgent()

        # 1. Update Price (async)
        response = asyncio.run(agent.execute("Update TEST_SYM to 999.99"))
        self.assertIn("Updated TEST_SYM", response)

        # Verify in manager
        mgr = MarketDataManager()
        price = mgr.get_price("TEST_SYM")
        self.assertEqual(price, 999.99)

    def test_news_injection(self):
        agent = MarketUpdateAgent()
        headline = "Adam AI takes over the world"
        response = asyncio.run(agent.execute(f"News: {headline}"))
        self.assertIn("Injected News", response)

        # Verify in file
        with open("showcase/js/market_snapshot.js", "r") as f:
            content = f.read()
            self.assertIn(headline, content)

    def test_scenario_activation(self):
        agent = MarketUpdateAgent()
        # Activate
        response = asyncio.run(agent.execute("Activate scenario BULL_RALLY"))
        self.assertIn("Active Scenario set to: Bull Rally", response)

        # Verify
        mgr = MarketDataManager()
        self.assertEqual(mgr.active_scenario.name, "Bull Rally")

        # Simulate
        response = asyncio.run(agent.execute("Simulate market"))
        self.assertIn("Bull Rally", response)

if __name__ == "__main__":
    unittest.main()
