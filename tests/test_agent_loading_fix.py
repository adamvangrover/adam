
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import logging
import importlib

# Ensure the project root is in the python path
sys.path.append(os.getcwd())

class TestAgentLoadingBug(unittest.TestCase):
    def setUp(self):
        # Create a dictionary of mocks to patch into sys.modules
        self.modules_to_patch = {}

        # Mock missing dependencies
        self.modules_to_patch['textblob'] = MagicMock()
        self.modules_to_patch['core.data_sources.financial_news_api'] = MagicMock()
        self.modules_to_patch['financial_digital_twin.nexus_agent'] = MagicMock()

        # Mock all agent modules
        agent_modules = [
            'market_sentiment_agent', 'macroeconomic_analysis_agent', 'geopolitical_risk_agent',
            'industry_specialist_agent', 'fundamental_analyst_agent', 'technical_analyst_agent',
            'risk_assessment_agent', 'newsletter_layout_specialist_agent', 'data_verification_agent',
            'lexica_agent', 'archive_manager_agent', 'agent_forge', 'prompt_tuner', 'code_alchemist',
            'lingua_maestro', 'sense_weaver', 'snc_analyst_agent', 'behavioral_economics_agent',
            'meta_cognitive_agent', 'NewsBot'
        ]

        for agent in agent_modules:
            mock_module = MagicMock()
            MockAgentClass = MagicMock()
            MockAgentClass.return_value = MagicMock()
            self.modules_to_patch[f'core.agents.{agent}'] = mock_module

        # Specifically set up MarketSentimentAgent
        mock_ms_module = MagicMock()
        mock_ms_class = MagicMock()
        setattr(mock_ms_module, 'MarketSentimentAgent', mock_ms_class)
        self.modules_to_patch['core.agents.market_sentiment_agent'] = mock_ms_module

    @patch('core.system.agent_orchestrator.MessageBroker')
    @patch('core.system.agent_orchestrator.load_config')
    @patch('core.system.agent_orchestrator.get_api_key')
    @patch('core.system.agent_orchestrator.LLMPlugin')
    def test_agent_loading_success(self, mock_llm_plugin, mock_get_api_key, mock_load_config, mock_message_broker):
        # Apply patches to sys.modules
        with patch.dict(sys.modules, self.modules_to_patch):
            # Import AgentOrchestrator INSIDE the test to ensure it uses the patched modules
            # We might need to reload it if it was already imported
            import core.system.agent_orchestrator
            importlib.reload(core.system.agent_orchestrator)
            from core.system.agent_orchestrator import AgentOrchestrator

            # Setup mocks
            mock_load_config.return_value = {}  # Empty config initially
            mock_get_api_key.return_value = "fake_key"

            # Mock MessageBroker.get_instance()
            mock_broker_instance = MagicMock()
            mock_message_broker.get_instance.return_value = mock_broker_instance

            orchestrator = AgentOrchestrator()

            # Inject a config that should trigger loading of MarketSentimentAgent
            orchestrator.config = {
                "agents": [
                    {
                        "name": "MarketSentimentAgent",
                        "description": "Analyzes market sentiment."
                    }
                ]
            }

            # Call load_agents
            orchestrator.load_agents()

            # Verify that the agent was loaded successfully
            self.assertIn("MarketSentimentAgent", orchestrator.agents)


if __name__ == '__main__':
    unittest.main()
