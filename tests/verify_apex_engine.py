import sys
import os
import unittest
from unittest.mock import MagicMock

# Add root to path
sys.path.append(os.getcwd())

from core.engine.states import init_risk_state
from core.engine.cyclical_reasoning_graph import cyclical_reasoning_app

class TestApexGraph(unittest.TestCase):
    def test_graph_execution(self):
        # Initialize state
        initial_state = init_risk_state("AAPL", "Assess credit risk")

        # Configure the graph to allow recursion if needed (recursion limit)
        config = {"recursion_limit": 20, "configurable": {"thread_id": "test_thread"}}

        # Invoke the graph
        print("Invoking Cyclical Reasoning Graph...")
        final_state = cyclical_reasoning_app.invoke(initial_state, config=config)

        print("Final State Status:", final_state["human_readable_status"])
        print("Critique Notes:", final_state["critique_notes"])

        self.assertTrue(len(final_state["critique_notes"]) > 0)
        self.assertIn("Score:", final_state["human_readable_status"])

if __name__ == "__main__":
    unittest.main()
