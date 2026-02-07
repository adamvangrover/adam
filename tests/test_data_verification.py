import unittest
import os
import json

class TestDataVerification(unittest.TestCase):
    def setUp(self):
        self.scenarios_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core', 'data', 'scenarios.json')
        self.report_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core', 'data', 'reports', '2026_01_30_strategic_equity_outlook.md')

    def test_scenarios_json_validity(self):
        """Verify scenarios.json is valid JSON and follows schema."""
        self.assertTrue(os.path.exists(self.scenarios_path), "scenarios.json not found")
        with open(self.scenarios_path, 'r') as f:
            data = json.load(f)
            self.assertIsInstance(data, list, "Scenarios root should be a list")
            for scenario in data:
                self.assertIn('id', scenario, "Scenario missing 'id'")
                self.assertIn('name', scenario, "Scenario missing 'name'")
                self.assertIn('shocks', scenario, "Scenario missing 'shocks'")
                self.assertIsInstance(scenario['shocks'], dict, "Shocks should be a dict")

    def test_bifurcated_scenario_exists(self):
        """Verify the specific 'Bifurcated Normalization' scenario exists."""
        with open(self.scenarios_path, 'r') as f:
            data = json.load(f)

        target = next((s for s in data if s['id'] == 'SCENARIO_BIFURCATED_NORMALIZATION'), None)
        self.assertIsNotNone(target, "SCENARIO_BIFURCATED_NORMALIZATION not found in scenarios.json")

        # Verify specific shocks from the thesis
        shocks = target['shocks']
        self.assertGreater(shocks.get('Technology', 0), 0, "Technology should be positive (Boom)")
        self.assertGreater(shocks.get('Materials', 0), 0, "Materials should be positive")

    def test_report_existence(self):
        """Verify the markdown report exists and has content."""
        self.assertTrue(os.path.exists(self.report_path), "Report file not found")
        with open(self.report_path, 'r') as f:
            content = f.read()
            self.assertGreater(len(content), 100, "Report content seems too short")
            self.assertIn("Bifurcated Normalization", content, "Report title/theme missing")

if __name__ == '__main__':
    unittest.main()
