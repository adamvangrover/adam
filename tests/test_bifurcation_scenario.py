import unittest
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine.sector_impact_engine import SectorImpactEngine

class TestBifurcationScenario(unittest.TestCase):
    def setUp(self):
        self.engine = SectorImpactEngine()
        self.scenario_id = "SCENARIO_BIFURCATED_NORMALIZATION"

    def test_scenario_load(self):
        scenario = next((s for s in self.engine.scenarios if s['id'] == self.scenario_id), None)
        self.assertIsNotNone(scenario, "Scenario not found in engine")
        self.assertEqual(scenario['shocks']['Technology'], 0.5)

    def test_scenario_impact(self):
        # Create a mock portfolio
        portfolio = [
            {"name": "Nvidia", "sector": "Technology", "leverage": 2.0},
            {"name": "LegacyCo", "sector": "Technology", "leverage": 4.0},
            {"name": "GoldMiner", "sector": "Materials", "leverage": 3.0}
        ]

        results = self.engine.analyze_portfolio(portfolio, scenario_id=self.scenario_id)

        # Check Tech impact (Should be positive boost)
        tech_res = next(r for r in results['results'] if r['asset'] == "Nvidia")
        self.assertIn("POSITIVE BOOST", tech_res['macro_insight'])

        # Check Materials impact
        mat_res = next(r for r in results['results'] if r['asset'] == "GoldMiner")
        self.assertIn("POSITIVE BOOST", mat_res['macro_insight'])

        print("\nSimulation Log:")
        print(json.dumps(results['simulation_log'], indent=2))

if __name__ == '__main__':
    unittest.main()
