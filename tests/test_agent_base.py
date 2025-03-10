# tests/test_agent_base.py
import unittest
from core.agents.agent_base import AgentBase

class MockAgent(AgentBase):  # Concrete subclass for testing
    def execute(self, *args, **kwargs):
        return "Mock Agent Executed"

class TestAgentBase(unittest.TestCase):
    def test_init_attributes(self):
        config = {"agent_name": "TestAgent", "description": "Test Description"}
        agent = MockAgent(config=config)
        self.assertEqual(agent.agent_name, "TestAgent")
        self.assertEqual(agent.description, "Test Description")
        self.assertEqual(agent.config, config) # Ensure the config is also stored

if __name__ == '__main__':
    unittest.main()
