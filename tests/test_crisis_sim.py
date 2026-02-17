import unittest
from scripts.market_mayhem_crisis_sim import CrisisSimulator

class TestCrisisSimulator(unittest.TestCase):
    def test_scenarios_exist(self):
        sim = CrisisSimulator("test")
        self.assertIn("2000_DOTCOM", sim.scenarios)
        self.assertIn("2022_INFLATION", sim.scenarios)
        self.assertIn("2025_AI_SHOCK", sim.scenarios)

    def test_run_simulation_returns_correct_structure(self):
        sim = CrisisSimulator("test")
        result = sim.run_simulation("2025_AI_SHOCK")
        self.assertIsNotNone(result)
        self.assertIn("scenario", result)
        self.assertIn("labels", result)
        self.assertIn("market", result)
        self.assertIn("portfolio", result)
        self.assertEqual(len(result["labels"]), len(result["market"]))
        self.assertEqual(len(result["labels"]), len(result["portfolio"]))

    def test_run_all_scenarios(self):
        sim = CrisisSimulator("test")
        results = sim.run_all_scenarios()
        self.assertEqual(len(results), 6)

if __name__ == '__main__':
    unittest.main()
