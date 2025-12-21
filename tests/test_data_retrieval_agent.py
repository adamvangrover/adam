# tests/test_data_retrieval_agent.py

import unittest
from unittest.mock import patch, mock_open
from core.agents.data_retrieval_agent import DataRetrievalAgent
from core.system.error_handler import DataNotFoundError, FileReadError, InvalidInputError
from core.system.knowledge_base import KnowledgeBase


class TestDataRetrievalAgent(unittest.TestCase):

    def setUp(self):
        # Mock config for testing.  Avoids needing real config files.
        self.mock_config = {
            "agents": {
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
        self.mock_kb = KnowledgeBase()  # Use the actual class for testing
        self.mock_kb.update("test_key", "Test Value")  # Add some data

    @patch('core.utils.config_utils.load_config')
    @patch('core.utils.data_utils.load_data')  # Mock load_data
    def test_get_risk_rating_found(self, mock_load_data, mock_load_config):
        mock_load_config.side_effect = [self.mock_config, self.mock_data_sources]  # return config then data_source
        mock_load_data.return_value = {"ABC": "low", "DEF": "medium"}  # Mocked data
        agent = DataRetrievalAgent(config=self.mock_config)
        rating = agent.get_risk_rating("ABC")
        self.assertEqual(rating, "low")
        mock_load_data.assert_called_once_with(self.mock_data_sources['risk_ratings'])

    @patch('core.utils.config_utils.load_config')
    @patch('core.utils.data_utils.load_data')
    def test_get_risk_rating_not_found(self, mock_load_data, mock_load_config):
        mock_load_config.side_effect = [self.mock_config, self.mock_data_sources]  # return config then data_source
        mock_load_data.return_value = {"ABC": "low"}  # Mocked data
        agent = DataRetrievalAgent(config=self.mock_config)
        with self.assertRaises(DataNotFoundError) as context:
            agent.get_risk_rating("XYZ")
        self.assertEqual(context.exception.code, 101)  # verify correct error
        self.assertIn("XYZ", context.exception.message)  # verify data identifier in message

    @patch('core.utils.config_utils.load_config')
    @patch('core.utils.data_utils.load_data')
    def test_get_risk_rating_file_not_found(self, mock_load_data, mock_load_config):
        mock_load_config.side_effect = [self.mock_config, self.mock_data_sources]  # return config then data_source
        mock_load_data.return_value = None
        agent = DataRetrievalAgent(config=self.mock_config)
        with self.assertRaises(FileReadError) as context:
            agent.get_risk_rating("ABC")
        self.assertEqual(context.exception.code, 105)

    @patch('core.utils.config_utils.load_config')
    @patch('core.utils.data_utils.load_data')
    def test_get_market_data(self, mock_load_data, mock_load_config):
        mock_load_config.side_effect = [self.mock_config, self.mock_data_sources]  # return config then data_source
        mock_load_data.return_value = {"market_trends": ["test"]}
        agent = DataRetrievalAgent(config=self.mock_config)
        result = agent.get_market_data()
        self.assertEqual(result, {"market_trends": ["test"]})
        mock_load_data.assert_called_once_with(self.mock_data_sources['market_baseline'])

    @patch('core.utils.config_utils.load_config')
    @patch('core.utils.data_utils.load_data')
    def test_execute_risk_rating(self, mock_load_data, mock_load_config):
        mock_load_config.side_effect = [self.mock_config, self.mock_data_sources]  # return config then data_source
        mock_load_data.return_value = {"ABC": "low"}
        agent = DataRetrievalAgent(config=self.mock_config)
        result = agent.execute("risk_rating:ABC")
        self.assertEqual(result, "The risk rating for ABC is low.")
        mock_load_data.assert_called()

    @patch('core.utils.config_utils.load_config')
    def test_execute_kb_query(self, mock_load_config):
        mock_load_config.side_effect = [self.mock_config, {}]
        agent = DataRetrievalAgent(config={'knowledge_base': self.mock_kb})
        result = agent.execute("kb:test_key")
        self.assertEqual(result, "Test Value")

    @patch('core.utils.config_utils.load_config')
    def test_execute_invalid_command(self, mock_load_config):
        mock_load_config.side_effect = [self.mock_config, {}]
        agent = DataRetrievalAgent(config=self.mock_config)
        with self.assertRaises(InvalidInputError) as context:
            agent.execute("invalid command")
        self.assertEqual(context.exception.code, 103)


if __name__ == '__main__':
    unittest.main()
