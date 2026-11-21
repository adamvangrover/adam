# tests/test_agent_orchestrator.py

import sys
import unittest
from unittest.mock import patch, MagicMock
from core.system.agent_orchestrator import AgentOrchestrator
from core.agents.agent_base import AgentBase
from core.system.error_handler import AgentNotFoundError
from core.system import agent_orchestrator as ao_module


class MockAgent(AgentBase):  # Define a mock agent for testing
    async def execute(self, *args, **kwargs):
        return "Mock Agent Result"


class TestAgentOrchestrator(unittest.TestCase):

    def setUp(self):
        self.llm_patcher = patch('core.system.agent_orchestrator.LLMPlugin')
        self.mock_llm_plugin = self.llm_patcher.start()

        # Patch AGENT_CLASSES
        self.original_agent_classes = ao_module.AGENT_CLASSES.copy()
        ao_module.AGENT_CLASSES["MockAgent"] = MockAgent

        self.mock_config = {
            "agents": {
                "MockAgent": {
                    "class": "tests.test_agent_orchestrator.MockAgent",  # Use the full path
                    "arguments": {"config": {}}  # Pass in the required arguments
                }
            }
        }
        self.test_config = {"test": "test"}

    def tearDown(self):
        self.llm_patcher.stop()
        ao_module.AGENT_CLASSES = self.original_agent_classes

    @patch('core.utils.config_utils.load_config')
    def test_load_agents(self, mock_load_config):
        mock_load_config.return_value = self.mock_config
        orchestrator = AgentOrchestrator()
        # Mocking config directly on instance to avoid __init__ reload issues
        orchestrator.config = {"agents": self.mock_config["agents"]}
        orchestrator.load_agents()
        self.assertIn("MockAgent", orchestrator.agents)
        self.assertIsInstance(orchestrator.agents["MockAgent"], MockAgent)

    @patch('core.utils.config_utils.load_config')
    def test_load_agents_invalid_class(self, mock_load_config):
        mock_load_config.return_value = {
            "agents": {"InvalidAgent": {"class": "nonexistent.module.Class", "arguments": {}}}
        }
        orchestrator = AgentOrchestrator()
        orchestrator.config = {"agents": {"InvalidAgent": {"class": "nonexistent.module.Class", "arguments": {}}}}
        # AgentOrchestrator handles load errors by logging, it does NOT raise ImportError usually.
        # But let's check implementation.
        # try: ... except Exception as e: logging.error(...)
        # So it won't raise.
        # We should check that it is NOT loaded.
        orchestrator.load_agents()
        self.assertNotIn("InvalidAgent", orchestrator.agents)

    @patch('core.utils.config_utils.load_config')
    def test_get_agent_found(self, mock_load_config):
        mock_load_config.return_value = self.mock_config
        orchestrator = AgentOrchestrator()
        orchestrator.config = {"agents": self.mock_config["agents"]}
        orchestrator.load_agents()
        agent = orchestrator.get_agent("MockAgent")
        self.assertIsInstance(agent, MockAgent)

    @patch('core.utils.config_utils.load_config')
    def test_get_agent_not_found(self, mock_load_config):
        mock_load_config.return_value = self.mock_config
        orchestrator = AgentOrchestrator()
        orchestrator.config = {"agents": self.mock_config["agents"]}
        orchestrator.load_agents()
        agent = orchestrator.get_agent("NonexistentAgent")
        self.assertIsNone(agent)

    @patch('core.utils.config_utils.load_config')
    def test_execute_agent_success(self, mock_load_config):
        # execute_agent is async-fire-and-forget (returns None)
        mock_load_config.return_value = self.mock_config
        orchestrator = AgentOrchestrator()
        orchestrator.config = {"agents": self.mock_config["agents"]}
        orchestrator.load_agents()

        orchestrator.message_broker = MagicMock()

        result = orchestrator.execute_agent("MockAgent", context={})
        self.assertIsNone(result)
        orchestrator.message_broker.publish.assert_called_once()

    @patch('core.utils.config_utils.load_config')
    def test_execute_agent_not_found(self, mock_load_config):
        mock_load_config.return_value = self.mock_config
        orchestrator = AgentOrchestrator()
        orchestrator.config = {"agents": self.mock_config["agents"]}
        orchestrator.load_agents()

        result = orchestrator.execute_agent("NonexistentAgent", context={})
        self.assertIsNone(result)
        # Should log error, but return None.

    # Remove test_execute_agent_exception as execute_agent catches exceptions and logs them


if __name__ == '__main__':
    unittest.main()
