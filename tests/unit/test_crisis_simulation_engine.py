import unittest
import networkx as nx
from core.engine.simulation_engine import CrisisSimulationEngine

class TestCrisisSimulationEngine(unittest.TestCase):
    def setUp(self):
        self.kg = nx.DiGraph()
        self.engine = CrisisSimulationEngine(self.kg)

        # Sample markdown content based on the real file structure
        self.sample_markdown = """# PROMPT: Test Scenario
**ID:** TEST-001
**Tags:** [Test, Simulation]

## Scenario
**Date:** 2026-01-01
**Event:** "Test Event"
Description here.
**Market Impact:**
*   **Indices:** SPX -5%, NDX -3%
*   **Sectors:** Tech -10%

## Task
Do something.
"""

    def test_extract_title(self):
        title = self.engine._extract_title(self.sample_markdown)
        self.assertEqual(title, "Test Scenario")

    def test_extract_field_id(self):
        id_val = self.engine._extract_field(self.sample_markdown, "ID")
        self.assertEqual(id_val, "TEST-001")

    def test_extract_list_tags(self):
        tags = self.engine._extract_list(self.sample_markdown, "Tags")
        self.assertEqual(tags, ["Test", "Simulation"])

    def test_parse_shocks(self):
        shocks = self.engine._parse_shocks(self.sample_markdown)
        # "Tech -10%" is ignored because it's not all CAPS.
        self.assertEqual(len(shocks), 2)

        self.assertEqual(shocks[0]['target'], "SPX")
        self.assertEqual(shocks[0]['change'], "-5%")
        self.assertEqual(shocks[1]['target'], "NDX")
        self.assertEqual(shocks[1]['change'], "-3%")

    def test_parse_shocks_real_behavior(self):
        # Let's test with content that matches the regex assumptions (All Caps Tickers)
        content = """# PROMPT: Test
**Market Impact:**
* **Indices:** SPX -5%
"""
        shocks = self.engine._parse_shocks(content)
        self.assertEqual(shocks[0]['target'], 'SPX')
        self.assertEqual(shocks[0]['change'], '-5%')

if __name__ == '__main__':
    unittest.main()
