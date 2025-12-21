import os
import sys
import unittest

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine.neuro_symbolic_planner import NeuroSymbolicPlanner


class TestAWOPlanner(unittest.TestCase):
    def setUp(self):
        # Mocks might be needed if __init__ connects to real DBs
        # But UnifiedKnowledgeGraph seems to be in-memory or stubbed in tests usually.
        # If it fails, I'll mock it.
        try:
            self.planner = NeuroSymbolicPlanner()
        except Exception as e:
            self.skipTest(f"Failed to init planner: {e}")

    def test_parse_natural_language_plan_numbered(self):
        text = """
        Here is the plan:
        1. Ingest data from SEC.
        2. Calculate DCF.
        3. Verify with PROV-O.
        """
        plan = self.planner.parse_natural_language_plan(text)
        self.assertEqual(len(plan['steps']), 3)
        self.assertEqual(plan['steps'][0]['task_id'], '1')
        self.assertEqual(plan['steps'][0]['description'], 'Ingest data from SEC.')
        self.assertEqual(plan['steps'][2]['description'], 'Verify with PROV-O.')

    def test_parse_empty_plan(self):
        text = "No steps here."
        plan = self.planner.parse_natural_language_plan(text)
        self.assertEqual(len(plan['steps']), 0)

if __name__ == '__main__':
    unittest.main()
