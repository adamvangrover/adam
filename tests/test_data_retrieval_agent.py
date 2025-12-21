# tests/test_data_retrieval_agent.py

import unittest
from unittest.mock import patch, mock_open
from core.agents.data_retrieval_agent import DataRetrievalAgent
from core.system.error_handler import DataNotFoundError, FileReadError, InvalidInputError
from core.system.knowledge_base import KnowledgeBase

class TestDataRetrievalAgent(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Mock config for testing.  Avoids needing real config files.
        self.mock_config = {
            "agents":{
                "DataRetrievalAgent": {
                "persona": "Test Persona",
                "description": "Test Description",
                "expertise": ["data access"]
                }
            }
        }
        self.mock_data_sources = {
            "risk_ratings": {"type": "json", "path": "dummy_path.json"},
             "market_baseline": {"type": "json", "path": "dummy_path_market.json"}
        }
        self.mock_kb = KnowledgeBase() # Use the actual class for testing
        self.mock_kb.update("test_key", "Test Value") # Add some data

    @patch('core.utils.config_utils.load_config')
    @patch('core.agents.data_retrieval_agent.load_data') # Mock load_data patched where used
    def test_get_risk_rating_found(self, mock_load_data, mock_load_config):
        mock_load_config.side_effect = [self.mock_config, self.mock_data_sources] #return config then data_source
        mock_load_data.return_value = {"ABC": "low", "DEF": "medium"}  # Mocked data
        agent = DataRetrievalAgent(config=self.mock_config)
        rating = agent.get_risk_rating("ABC")
        self.assertEqual(rating, "low")

    @patch('core.utils.config_utils.load_config')
    @patch('core.agents.data_retrieval_agent.load_data')
    def test_get_risk_rating_not_found(self, mock_load_data, mock_load_config):
        mock_load_config.side_effect = [self.mock_config, self.mock_data_sources] #return config then data_source
        mock_load_data.return_value = {"ABC": "low"}  # Mocked data
        agent = DataRetrievalAgent(config=self.mock_config)
        # The agent logs warning and returns None, does not raise DataNotFoundError in current impl
        rating = agent.get_risk_rating("XYZ")
        self.assertIsNone(rating)

    @patch('core.utils.config_utils.load_config')
    @patch('core.agents.data_retrieval_agent.load_data')
    def test_get_risk_rating_file_not_found(self, mock_load_data, mock_load_config):
        mock_load_config.side_effect = [self.mock_config, self.mock_data_sources] #return config then data_source
        # Raise FileReadError from load_data
        mock_load_data.side_effect = FileReadError("path", "msg")
        agent = DataRetrievalAgent(config=self.mock_config)

        # Current impl catches FileReadError and returns None
        rating = agent.get_risk_rating("ABC")
        self.assertIsNone(rating)

    @patch('core.utils.config_utils.load_config')
    @patch('core.agents.data_retrieval_agent.load_data')
    def test_get_market_data(self, mock_load_data, mock_load_config):
        mock_load_config.side_effect = [self.mock_config, self.mock_data_sources]  # return config then data_source
        mock_load_data.return_value = {"market_trends":["test"]}
        agent = DataRetrievalAgent(config=self.mock_config)
        result = agent.get_market_data()
        self.assertEqual(result, {"market_trends":["test"]})


    @patch('core.utils.config_utils.load_config')
    @patch('core.agents.data_retrieval_agent.load_data')
    async def test_execute_risk_rating(self, mock_load_data, mock_load_config):
        mock_load_config.side_effect = [self.mock_config, self.mock_data_sources] #return config then data_source
        mock_load_data.return_value = {"ABC": "low"}
        agent = DataRetrievalAgent(config=self.mock_config)
        result = await agent.execute({'data_type': 'get_risk_rating', 'company_id': 'ABC'})
        self.assertEqual(result, "low")
        mock_load_data.assert_called()

    @patch('core.utils.config_utils.load_config')
    async def test_execute_kb_query(self, mock_load_config):
        mock_load_config.side_effect = [self.mock_config, {}]
        agent = DataRetrievalAgent(config={'knowledge_base': self.mock_kb})
        result = await agent.execute({'data_type': 'access_knowledge_base', 'query': 'test_key'})
        self.assertEqual(result, "Test Value")

    @patch('core.utils.config_utils.load_config')
    async def test_execute_invalid_command(self, mock_load_config):
        mock_load_config.side_effect = [self.mock_config, {}]
        agent = DataRetrievalAgent(config=self.mock_config)
        # Invalid input currently returns None and logs warning, doesn't raise InvalidInputError
        result = await agent.execute({'data_type': 'invalid command'})
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
