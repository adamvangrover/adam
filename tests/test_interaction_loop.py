# tests/test_interaction_loop.py

import unittest
from unittest.mock import patch, MagicMock
from core.system.interaction_loop import InteractionLoop
from core.system.agent_orchestrator import AgentOrchestrator  # Import
from core.agents.query_understanding_agent import QueryUnderstandingAgent
from core.agents.data_retrieval_agent import DataRetrievalAgent
from core.agents.result_aggregation_agent import ResultAggregationAgent
from core.system.error_handler import InvalidInputError, AgentNotFoundError, DataNotFoundError
from core.system.knowledge_base import KnowledgeBase

class TestInteractionLoop(unittest.TestCase):

    def setUp(self):
        self.mock_config = {
            "interaction_loop": {
                "max_iterations": 5,
                "reprompt_strategy": "retry"
             },
            "agents": {
                "QueryUnderstandingAgent": {}, "DataRetrievalAgent": {}, "ResultAggregationAgent": {}
            }
        }
        self.mock_orchestrator = MagicMock(spec=AgentOrchestrator)
        self.mock_query_agent = MagicMock(spec=QueryUnderstandingAgent)
        self.mock_data_agent = MagicMock(spec=DataRetrievalAgent)
        self.mock_result_agent = MagicMock(spec=ResultAggregationAgent)
        self.mock_kb = MagicMock(spec=KnowledgeBase)


        # Set up the return values for the mocked agents
        self.mock_orchestrator.get_agent.side_effect = lambda x: {
            "QueryUnderstandingAgent": self.mock_query_agent,
            "DataRetrievalAgent": self.mock_data_agent,
            "ResultAggregationAgent": self.mock_result_agent,
        }.get(x)

    @patch('core.utils.config_utils.load_config')
    @patch('core.system.agent_orchestrator.AgentOrchestrator', return_value=MagicMock(spec=AgentOrchestrator))
    def test_process_input_risk_query(self, mock_orchestrator, mock_config):
        mock_config.return_value = self.mock_config
        loop = InteractionLoop(config=self.mock_config, knowledge_base = self.mock_kb) #Pass Knowledge Base

        # Mock the agent methods for this specific test case
        self.mock_query_agent.execute.return_value = ["DataRetrievalAgent"]
        self.mock_data_agent.execute.return_value = "low"
        self.mock_result_agent.execute.return_value = "The risk rating is low."
        mock_orchestrator.return_value.execute_agent = MagicMock(side_effect = ["DataRetrievalAgent", "low"]) # Orchestrator

        result = loop.process_input("risk ABC")
        self.assertEqual(result, "The risk rating is low.")
        self.mock_query_agent.execute.assert_called_once_with("risk ABC")
        mock_orchestrator.return_value.execute_agent.assert_any_call("DataRetrievalAgent", "risk_rating:ABC") # Check orchestrator call
        self.mock_result_agent.execute.assert_called_once_with(["low"]) # Result agent called with data agent result

    @patch('core.utils.config_utils.load_config')
    @patch('core.system.agent_orchestrator.AgentOrchestrator', return_value=MagicMock(spec=AgentOrchestrator))
    def test_process_input_kb_query(self, mock_orchestrator, mock_config):
        mock_config.return_value = self.mock_config
        loop = InteractionLoop(config=self.mock_config, knowledge_base = self.mock_kb)

        self.mock_query_agent.execute.return_value = ["DataRetrievalAgent"]
        self.mock_data_agent.execute.return_value = "Positive"
        self.mock_result_agent.execute.return_value = "The market sentiment is Positive."
        mock_orchestrator.return_value.execute_agent = MagicMock(side_effect = ["DataRetrievalAgent", "Positive"])


        result = loop.process_input("kb:market_sentiment")
        self.assertEqual(result, "The market sentiment is Positive.")
        self.mock_query_agent.execute.assert_called_once_with("kb:market_sentiment")
        mock_orchestrator.return_value.execute_agent.assert_any_call("DataRetrievalAgent", "kb:market_sentiment")
        self.mock_result_agent.execute.assert_called_once_with(["Positive"])

    @patch('core.utils.config_utils.load_config')
    @patch('core.system.agent_orchestrator.AgentOrchestrator', return_value=MagicMock(spec=AgentOrchestrator))
    def test_process_input_updatekb(self, mock_orchestrator, mock_config):
        mock_config.return_value = self.mock_config
        loop = InteractionLoop(config=self.mock_config, knowledge_base=self.mock_kb)
        self.mock_query_agent.execute.return_value = [] # No agents needed
        mock_orchestrator.return_value.execute_agent = MagicMock(return_value = [])
        result = loop.process_input("updatekb test_key:test_value")
        self.assertEqual(result, "Knowledge base updated.")
        self.mock_kb.update.assert_called_once_with("test_key", "test_value")

    @patch('core.utils.config_utils.load_config')
    @patch('core.system.agent_orchestrator.AgentOrchestrator', return_value=MagicMock(spec=AgentOrchestrator))
    def test_process_input_invalid_command(self, mock_orchestrator, mock_config):
        mock_config.return_value = self.mock_config
        loop = InteractionLoop(config=self.mock_config, knowledge_base = self.mock_kb)
        self.mock_query_agent.execute.side_effect = InvalidInputError("invalid", "Test Reason")
        mock_orchestrator.return_value.execute_agent = MagicMock(side_effect = InvalidInputError("invalid", "Test Reason"))

        with self.assertRaises(InvalidInputError) as context:
            loop.process_input("invalid command")
        self.assertEqual(context.exception.code, 103)
        self.mock_query_agent.execute.assert_called_once_with("invalid command")


    @patch('core.utils.config_utils.load_config')
    @patch('core.system.agent_orchestrator.AgentOrchestrator', return_value=MagicMock(spec=AgentOrchestrator))
    def test_process_input_agent_not_found(self, mock_orchestrator, mock_config):
        mock_config.return_value = self.mock_config
        loop = InteractionLoop(config=self.mock_config, knowledge_base = self.mock_kb)
        self.mock_query_agent.execute.return_value = ["NonexistentAgent"]
        mock_orchestrator.return_value.execute_agent.side_effect = AgentNotFoundError("NonexistentAgent")

        with self.assertRaises(AgentNotFoundError) as context:
            loop.process_input("some query")
        self.assertEqual(context.exception.code, 102)

    @patch('core.utils.config_utils.load_config')
    @patch('core.system.agent_orchestrator.AgentOrchestrator', return_value=MagicMock(spec=AgentOrchestrator))
    def test_process_input_data_not_found(self, mock_orchestrator, mock_config):
        mock_config.return_value = self.mock_config
        loop = InteractionLoop(config=self.mock_config, knowledge_base=self.mock_kb)
        self.mock_query_agent.execute.return_value = ["DataRetrievalAgent"]
        mock_orchestrator.return_value.execute_agent.side_effect = DataNotFoundError()
        with self.assertRaises(DataNotFoundError) as context:
            loop.process_input("risk ABC")
        self.assertEqual(context.exception.code, 101)

    @patch('core.utils.config_utils.load_config')
    @patch('core.system.agent_orchestrator.AgentOrchestrator', return_value=MagicMock(spec=AgentOrchestrator))
    def test_process_input_multiple_agents(self, mock_orchestrator, mock_config):
        mock_config.return_value = self.mock_config
        loop = InteractionLoop(config=self.mock_config, knowledge_base = self.mock_kb)
        self.mock_query_agent.execute.return_value = ["Agent1", "Agent2"]
        mock_orchestrator.return_value.execute_agent.side_effect = ["Result1", "Result2"] # Orchestrator returns
        self.mock_result_agent.execute.return_value = "Combined Result"
        result = loop.process_input("some query")
        self.assertEqual(result, "Combined Result")
        self.assertEqual(mock_orchestrator.return_value.execute_agent.call_count, 2) # Two calls

if __name__ == '__main__':
    unittest.main()
