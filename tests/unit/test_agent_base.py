# tests/test_agent_base.py
import unittest
import asyncio
from core.agents.agent_base import AgentBase


class MockAgent(AgentBase):  # Concrete subclass for testing
    def __init__(self, config):
        super().__init__(config)
        # self.name is a property in base class

    async def execute(self, *args, **kwargs):
        return "Mock Agent Executed"


class TestAgentBase(unittest.TestCase):
    def test_init_attributes(self):
        config = {"agent_id": "TestAgent", "description": "Test Description"}
        agent = MockAgent(config=config)
        self.assertEqual(agent.name, "TestAgent")
        self.assertEqual(agent.config, config)


if __name__ == '__main__':
    unittest.main()
