#tests/test_agent_orchestrator.py

import unittest
from unittest.mock import patch, MagicMock
from core.system.agent_orchestrator import AgentOrchestrator
from core.agents.agent_base import AgentBase
from core.system.error_handler import AgentNotFoundError

class MockAgent(AgentBase):  # Define a mock agent for testing
    def execute(self, *args, **kwargs):
        return "Mock Agent Result"

class TestAgentOrchestrator(unittest.TestCase):

    def setUp(self):
        self.mock_config = {
            "agents": {
                "MockAgent": {
                    "class": "tests.test_agent_orchestrator.MockAgent",  # Use the full path
                    "arguments": {"config": {}} # Pass in the required arguments
                }
            }
        }
        self.test_config = {"test":"test"}

    @patch('core.utils.config_utils.load_config')
    def test_load_agents(self, mock_load_config):
        mock_load_config.return_value = self.mock_config
        orchestrator = AgentOrchestrator(config=self.test_config)
        orchestrator.load_agents()
        self.assertIn("MockAgent", orchestrator.agents)
        self.assertIsInstance(orchestrator.agents["MockAgent"], MockAgent)

    @patch('core.utils.config_utils.load_config')
    def test_load_agents_invalid_class(self, mock_load_config):
        mock_load_config.return_value = {
            "agents": { "InvalidAgent": {"class": "nonexistent.module.Class", "arguments": {}} }
        }
        orchestrator = AgentOrchestrator(config=self.test_config)
        with self.assertRaises(ImportError):  # Expect an ImportError
             orchestrator.load_agents()


    @patch('core.utils.config_utils.load_config')
    def test_get_agent_found(self, mock_load_config):
        mock_load_config.return_value = self.mock_config
        orchestrator = AgentOrchestrator(config=self.test_config)
        orchestrator.load_agents()
        agent = orchestrator.get_agent("MockAgent")
        self.assertIsInstance(agent, MockAgent)

    @patch('core.utils.config_utils.load_config')
    def test_get_agent_not_found(self, mock_load_config):
        mock_load_config.return_value = self.mock_config
        orchestrator = AgentOrchestrator(config=self.test_config)
        orchestrator.load_agents()
        agent = orchestrator.get_agent("NonexistentAgent")
        self.assertIsNone(agent)

    @patch('core.utils.config_utils.load_config')
    def test_execute_agent_success(self, mock_load_config):
        mock_load_config.return_value = self.mock_config
        orchestrator = AgentOrchestrator(config=self.test_config)
        orchestrator.load_agents()
        mock_agent = orchestrator.agents["MockAgent"]
        mock_agent.execute = MagicMock(return_value="Mocked Result") # Mock execute
        result = orchestrator.execute_agent("MockAgent")
        self.assertEqual(result, "Mocked Result")
        mock_agent.execute.assert_called_once() # Verify execute called


    @patch('core.utils.config_utils.load_config')
    def test_execute_agent_not_found(self, mock_load_config):
        mock_load_config.return_value = self.mock_config
        orchestrator = AgentOrchestrator(config=self.test_config)
        orchestrator.load_agents()  # Load agents (but we'll ask for one not loaded).
        with self.assertRaises(AgentNotFoundError) as context:
            orchestrator.execute_agent("NonexistentAgent")
        self.assertEqual(context.exception.code, 102)

    @patch('core.utils.config_utils.load_config')
    def test_execute_agent_exception(self, mock_load_config):
        mock_load_config.return_value = self.mock_config
        orchestrator = AgentOrchestrator(config=self.test_config)
        orchestrator.load_agents()
        mock_agent = orchestrator.agents["MockAgent"]
        mock_agent.execute = MagicMock(side_effect=Exception("Test Exception"))  # Simulate exception
        with self.assertRaises(Exception) as context:  # Catch the *general* exception
             orchestrator.execute_agent("MockAgent")
        self.assertEqual(str(context.exception), "Test Exception") # Verify exception message
        mock_agent.execute.assert_called_once()

if __name__ == '__main__':
    unittest.main()
