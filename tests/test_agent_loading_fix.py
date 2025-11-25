
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import logging

# Ensure the project root is in the python path
sys.path.append(os.getcwd())

# Mock missing dependencies and side-effects
sys.modules['textblob'] = MagicMock()
sys.modules['core.data_sources.financial_news_api'] = MagicMock()

# Mock all agent modules
agent_modules = [
    'market_sentiment_agent', 'macroeconomic_analysis_agent', 'geopolitical_risk_agent',
    'industry_specialist_agent', 'fundamental_analyst_agent', 'technical_analyst_agent',
    'risk_assessment_agent', 'newsletter_layout_specialist_agent', 'data_verification_agent',
    'lexica_agent', 'archive_manager_agent', 'agent_forge', 'prompt_tuner', 'code_alchemist',
    'lingua_maestro', 'sense_weaver', 'SNC_analyst_agent', 'behavioral_economics_agent',
    'meta_cognitive_agent', 'NewsBot'
]

for agent in agent_modules:
    mock_module = MagicMock()
    # Ensure the module has the agent class as an attribute
    # e.g. MarketSentimentAgent inside market_sentiment_agent module

    # We need to correctly map module name to class name based on AGENT_CLASSES or conventions
    # For now, let's assume class name = capitalized(agent_name) or just "MarketSentimentAgent" for market_sentiment_agent

    # Actually, AGENT_CLASSES maps "MarketSentimentAgent" -> "core.agents.market_sentiment_agent"
    # So the agent name "MarketSentimentAgent" is what we look for in the module.

    # Construct a Mock Class
    MockAgentClass = MagicMock()
    # make instance of it not fail
    MockAgentClass.return_value = MagicMock()

    # We need to set this class on the mocked module
    # But which name?
    # For 'market_sentiment_agent', the class is 'MarketSentimentAgent'
    # We can infer it from the list iteration or just hardcode for the one we test.

    sys.modules[f'core.agents.{agent}'] = mock_module

# Specifically set up MarketSentimentAgent for the test
mock_ms_module = sys.modules['core.agents.market_sentiment_agent']
mock_ms_class = MagicMock()
setattr(mock_ms_module, 'MarketSentimentAgent', mock_ms_class)

sys.modules['financial_digital_twin.nexus_agent'] = MagicMock()


from core.system.agent_orchestrator import AgentOrchestrator

class TestAgentLoadingBug(unittest.TestCase):
    @patch('core.system.agent_orchestrator.RabbitMQClient')
    @patch('core.system.agent_orchestrator.load_config')
    @patch('core.system.agent_orchestrator.get_api_key')
    @patch('core.system.agent_orchestrator.LLMPlugin') # Mock LLMPlugin
    def test_agent_loading_success(self, mock_llm_plugin, mock_get_api_key, mock_load_config, mock_rabbitmq):
        # Setup mocks
        mock_load_config.return_value = {} # Empty config initially
        mock_get_api_key.return_value = "fake_key"

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

        # Verify that the class was instantiated
        # We need to check if the mocked class was called
        # Since we patched imports after AgentOrchestrator import, relying on sys.modules might be tricky
        # because AgentOrchestrator might have already imported the real modules if they existed.
        # But we mocked them before AgentOrchestrator import (except textblob etc).
        # Actually, AgentOrchestrator imports happen at top level.
        # So sys.modules need to be set BEFORE `from core.system.agent_orchestrator import ...`

        # In this script, we did set sys.modules before import.

        # Since AgentOrchestrator imports MarketSentimentAgent at top level:
        # from core.agents.market_sentiment_agent import MarketSentimentAgent
        # It should pick up our mock if sys.modules was set.

        # However, the fix uses `globals()[agent_name]` first.
        # If AgentOrchestrator imported it, it's in its globals.
        # If the import succeeded (using our mock), then it should work.

if __name__ == '__main__':
    unittest.main()
