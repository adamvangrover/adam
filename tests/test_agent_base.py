# tests/test_agent_base.py
import unittest
import asyncio
from core.agents.agent_base import AgentBase


class MockAgent(AgentBase):  # Concrete subclass for testing
    def __init__(self, config):
        super().__init__(config)
        self.name = config.get("name")

    async def execute(self, *args, **kwargs):
        return "Mock Agent Executed"


class TestAgentBase(unittest.TestCase):
    def test_init_attributes(self):
        config = {"name": "TestAgent", "description": "Test Description"}
        agent = MockAgent(config=config)
        self.assertEqual(agent.name, "TestAgent")
        self.assertEqual(agent.config, config)  # Ensure the config is also stored


if __name__ == '__main__':
    unittest.main()
