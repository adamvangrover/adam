import unittest
import os
import yaml
from core.market_data.scenario_loader import loader
from core.market_data.scenarios import SCENARIOS, load_external_scenarios
from core.market_data.manager import MarketDataManager

class TestScenarioLoading(unittest.TestCase):

    def setUp(self):
        # Create a test scenario file
        self.test_file = "data/scenarios/test_crash.yaml"
        with open(self.test_file, "w") as f:
            f.write("""
name: Test Crash
description: A test scenario with a scheduled event.
global_drift: 0.0
global_volatility_multiplier: 1.0
scheduled_events:
  - step: 2
    symbol: TEST_SYM
    change: -0.50
    news: "Test Symbol Crashes 50%"
            """)

        # Ensure TEST_SYM exists in state for manager tests
        self.mgr = MarketDataManager()
        self.mgr.update_symbol("TEST_SYM", 100.0)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_load_scenarios(self):
        # Trigger reload
        load_external_scenarios()
        self.assertIn("TEST_CRASH", SCENARIOS)

        scenario = SCENARIOS["TEST_CRASH"]
        self.assertEqual(scenario.name, "Test Crash")
        self.assertEqual(len(scenario.scheduled_events), 1)
        self.assertEqual(scenario.scheduled_events[0].trigger_step, 2)

    def test_event_triggering(self):
        load_external_scenarios()
        self.mgr.set_scenario("TEST_CRASH")

        # Step 1: No change (drift is 0)
        self.mgr.simulate_step()
        price_1 = self.mgr.get_price("TEST_SYM")
        # Should be close to 100 (random shock is small)
        self.assertTrue(99 < price_1 < 101)

        # Step 2: Crash Event (-50%)
        self.mgr.simulate_step()
        price_2 = self.mgr.get_price("TEST_SYM")

        # Price should be roughly 50% of price_1
        # Note: drift is applied AFTER event in simulate_step loop for that symbol?
        # My code: Event logic updates symbol. Then standard loop updates it AGAIN.
        # This is fine, but checking exact value is tricky due to volatility.
        # But a 50% drop is huge.
        self.assertLess(price_2, 60.0)

        # Verify News
        latest_news = self.mgr.state["news_feed"][0]
        self.assertEqual(latest_news["headline"], "Test Symbol Crashes 50%")

if __name__ == "__main__":
    unittest.main()
