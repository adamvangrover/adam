import unittest
import asyncio
from core.agents.adaptive_agent import AdaptiveAgent

class TestAdaptiveAgent(unittest.TestCase):
    def setUp(self):
        self.config = {"agent_id": "test_adaptive_agent"}
        self.agent = AdaptiveAgent(self.config)

    def test_conviction_logic(self):
        """Test that the agent calculates conviction and switches modes."""
        # Case 1: Simple Task -> Direct
        mode, meta = self.agent.get_interaction_strategy("hi")
        self.assertEqual(mode, "DIRECT")

        # Case 2: Complex Task but defaulted conviction -> MCP
        # Complexity of "calculate optimize schema" is high (~0.3+)
        # Conviction defaults to 0.9. Adjusted = 0.9 - (0.3*0.2) = 0.84
        # Threshold is 0.85. So likely DIRECT fallback unless conviction is higher.

        # Let's force a high conviction history
        self.agent._conviction_history.append(1.0)
        mode, meta = self.agent.get_interaction_strategy("calculate optimize schema")
        # Complexity ~ 0.5 (keywords)
        # Adjusted = 1.0 - (0.5 * 0.2) = 0.9
        # 0.9 > 0.85 -> MCP
        self.assertEqual(mode, "MCP")

    def test_state_anchor(self):
        """Test anchor creation and drift detection."""
        anchor_id = self.agent.create_anchor()
        self.assertTrue(self.agent.verify_anchor(anchor_id))

        # Simulate drift
        self.agent.context["new_key"] = "drift"
        self.assertFalse(self.agent.verify_anchor(anchor_id))

    def test_tool_rag(self):
        """Test tool retrieval."""
        self.agent.register_tool_metadata("calc", "calculate numbers", lambda x: x)
        self.agent.register_tool_metadata("weather", "get weather info", lambda x: x)

        relevant = self.agent.retrieve_relevant_tools("calculate sum", limit=1)
        self.assertEqual(len(relevant), 1)
        self.assertEqual(relevant[0]["name"], "calc")

if __name__ == '__main__':
    unittest.main()
