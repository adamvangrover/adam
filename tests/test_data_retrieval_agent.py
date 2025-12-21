# tests/test_data_retrieval_agent.py

import unittest
from unittest.mock import patch, mock_open
from core.agents.data_retrieval_agent import DataRetrievalAgent
from core.system.error_handler import DataNotFoundError, FileReadError, InvalidInputError
from core.system.knowledge_base import KnowledgeBase


class TestDataRetrievalAgent(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Mock config for testing. Avoids needing real config files.
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
    @patch('core.agents.data_retrieval_agent.load_data') # Mock load_data patched where used
    def test_get_risk_rating_found(self, mock_load_data_agent, mock_load_data_utils, mock_load_config):
        # Note: Handling double patch for safety based on conflict context, 
        # but usually only one is needed depending on import style.
        mock_load_config.side_effect = [self.mock_config, self.mock_data_sources]
        mock_load_data_agent.return_value = {"ABC": "low", "DEF": "medium"} 
        
        agent = DataRetrievalAgent(config=self.mock_config)
        rating = agent.get_risk_rating("ABC")
        self.assertEqual(rating, "low")

    @patch('core.utils.config_utils.load_config')
    @patch('core.agents.data_retrieval_agent.load_data')
    def test_get_risk_rating_not_found(self, mock_load_data, mock_load_config):
        mock_load_config.side_effect = [self.mock_config, self.mock_data_sources]
        mock_load_data.return_value = {"ABC": "low"} 
        agent = DataRetrievalAgent(config=self.mock_config)
        
        # Adopted stricter error handling from fix-dependencies branch
        with self.assertRaises(DataNotFoundError) as context:
            agent.get_risk_rating("XYZ")
        self.assertEqual(context.exception.code, 101) 
        self.assertIn("XYZ", context.exception.message) 

    @patch('core.utils.config_utils.load_config')
    @patch('core.agents.data_retrieval_agent.load_data')
    def test_get_risk_rating_file_not_found(self, mock_load_data, mock_load_config):
        mock_load_config.side_effect = [self.mock_config, self.mock_data_sources]
        # Raise FileReadError from load_data (Main branch logic provides better simulation)
        mock_load_data.side_effect = FileReadError("path", "msg")
        
        agent = DataRetrievalAgent(config=self.mock_config)

        # Current impl catches FileReadError and returns None
        rating = agent.get_risk_rating("ABC")
        self.assertIsNone(rating)

    @patch('core.utils.config_utils.load_config')
    @patch('core.agents.data_retrieval_agent.load_data')
    def test_get_market_data(self, mock_load_data, mock_load_config):
        mock_load_config.side_effect = [self.mock_config, self.mock_data_sources]
        mock_load_data.return_value = {"market_trends": ["test"]}
        agent = DataRetrievalAgent(config=self.mock_config)
        result = agent.get_market_data()
        
        # Combined assertions: Check result AND check call args (stricter test)
        self.assertEqual(result, {"market_trends": ["test"]})
        mock_load_data.assert_called_once_with(self.mock_data_sources['market_baseline'])

    @patch('core.utils.config_utils.load_config')
    @patch('core.agents.data_retrieval_agent.load_data')
    async def test_execute_risk_rating(self, mock_load_data, mock_load_config):
        # Adopted async def from main branch
        mock_load_config.side_effect = [self.mock_config, self.mock_data_sources]
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
        
        # Hybrid merge: Stricter error checking (fix-branch) combined with async/await (main branch)
        with self.assertRaises(InvalidInputError) as context:
            await agent.execute({'data_type': 'invalid command'})
        self.assertEqual(context.exception.code, 103)

if __name__ == '__main__':
    unittest.main()