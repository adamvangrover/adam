# tests/test_system.py

from core.system.agent_orchestrator import AgentOrchestrator
import unittest
import yaml
import sys
import asyncio
from unittest.mock import patch, Mock, AsyncMock, MagicMock

# Patch heavy dependencies globally to avoid import-time costs/errors
sys.modules['spacy'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['tensorflow'] = MagicMock()
sys.modules['tweepy'] = MagicMock()
sys.modules['textblob'] = MagicMock()
sys.modules['semantic_kernel'] = MagicMock()
# sys.modules['langchain'] = MagicMock()
# sys.modules['langchain_community'] = MagicMock()

# Patch all agent modules to prevent heavy imports
agents = [
    'query_understanding_agent', 'data_retrieval_agent', 'market_sentiment_agent',
    'macroeconomic_analysis_agent', 'geopolitical_risk_agent', 'industry_specialist_agent',
    'fundamental_analyst_agent', 'technical_analyst_agent', 'risk_assessment_agent',
    'newsletter_layout_specialist_agent', 'data_verification_agent', 'lexica_agent',
    'archive_manager_agent', 'agent_forge', 'prompt_tuner', 'code_alchemist',
    'lingua_maestro', 'sense_weaver', 'snc_analyst_agent', 'behavioral_economics_agent',
    'meta_cognitive_agent', 'news_bot', 'report_generator_agent', 'result_aggregation_agent'
]
for agent in agents:
    sys.modules[f'core.agents.{agent}'] = MagicMock()
sys.modules['financial_digital_twin'] = MagicMock()
sys.modules['financial_digital_twin.nexus_agent'] = MagicMock()


class TestAgentOrchestrator(unittest.TestCase):
    def setUp(self):
        """Setup method to create an instance of AgentOrchestrator."""

        # Patch heavy dependencies within the orchestrator class usage
        self.llm_patcher = patch('core.system.agent_orchestrator.LLMPlugin')
        self.mock_llm = self.llm_patcher.start()

        self.broker_patcher = patch('core.system.agent_orchestrator.MessageBroker')
        self.mock_broker_cls = self.broker_patcher.start()
        self.mock_broker = self.mock_broker_cls.get_instance.return_value

        self.kernel_patcher = patch('core.system.agent_orchestrator.Kernel')
        self.mock_kernel = self.kernel_patcher.start()

        # Patch AGENT_CLASSES to avoid loading real agents during init or load_agents
        self.agent_classes_patcher = patch.dict('core.system.agent_orchestrator.AGENT_CLASSES', {}, clear=True)
        self.agent_classes_patcher.start()

        self.orchestrator = AgentOrchestrator()

    def tearDown(self):
        self.llm_patcher.stop()
        self.broker_patcher.stop()
        self.kernel_patcher.stop()
        self.agent_classes_patcher.stop()

    @patch('core.agents.market_sentiment_agent.MarketSentimentAgent.analyze_sentiment')
    @patch('core.agents.macroeconomic_analysis_agent.MacroeconomicAnalysisAgent.analyze_macroeconomic_data')
    def test_execute_workflow(self, mock_analyze_sentiment, mock_analyze_macroeconomic_data):
        """Test executing different workflows."""
        # Test the "generate_newsletter" workflow
        mock_analyze_sentiment.return_value = "positive"
        mock_analyze_macroeconomic_data.return_value = {'GDP_growth': 2.5, 'inflation': 2.0}

        self.orchestrator.workflows = {
            "generate_newsletter": {
                "agents": ["MarketSentimentAgent", "MacroeconomicAnalysisAgent"],
                "dependencies": {}
            }
        }

        mock_sentiment_agent = AsyncMock()
        mock_sentiment_agent.execute.return_value = "positive"
        mock_macro_agent = AsyncMock()
        mock_macro_agent.execute.return_value = {'GDP_growth': 2.5, 'inflation': 2.0}

        self.orchestrator.agents = {
            "MarketSentimentAgent": mock_sentiment_agent,
            "MacroeconomicAnalysisAgent": mock_macro_agent
        }

        result = asyncio.run(self.orchestrator.execute_workflow("generate_newsletter", initial_context={}))

        self.assertIsNotNone(result)
        self.assertIn("MarketSentimentAgent", result)
        self.assertIn("MacroeconomicAnalysisAgent", result)

    def test_agent_interactions(self):
        """Test interactions and data flow between agents."""
        mock_fund = AsyncMock()
        mock_tech = AsyncMock()
        mock_risk = AsyncMock()

        mock_fund.execute.return_value = {'some_data': 'some_value'}
        mock_tech.execute.return_value = {'other_data': 'other_value'}
        mock_risk.execute.return_value = {'risk_score': 0.5}

        self.orchestrator.agents = {
            "FundamentalAnalystAgent": mock_fund,
            "TechnicalAnalystAgent": mock_tech,
            "RiskAssessmentAgent": mock_risk
        }

        self.orchestrator.workflows = {
            "perform_company_analysis": {
                "agents": ["FundamentalAnalystAgent", "TechnicalAnalystAgent", "RiskAssessmentAgent"],
                "dependencies": {
                    "RiskAssessmentAgent": ["FundamentalAnalystAgent", "TechnicalAnalystAgent"]
                }
            }
        }

        asyncio.run(self.orchestrator.execute_workflow("perform_company_analysis",
                    initial_context={'company_data': {'name': 'Example Corp'}}))

        mock_risk.execute.assert_called_once()
        call_kwargs = mock_risk.execute.call_args.kwargs
        self.assertIn('FundamentalAnalystAgent', call_kwargs)
        self.assertEqual(call_kwargs['FundamentalAnalystAgent'], {'some_data': 'some_value'})


if __name__ == '__main__':
    unittest.main()
