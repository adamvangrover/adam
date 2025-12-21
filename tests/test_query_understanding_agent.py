# tests/test_query_understanding_agent.py

import unittest
from unittest.mock import patch

from core.agents.query_understanding_agent import QueryUnderstandingAgent
from core.system.error_handler import InvalidInputError


class TestQueryUnderstandingAgent(unittest.TestCase):

    @patch('core.agents.query_understanding_agent.LLMPlugin')
    def setUp(self, mock_llm_plugin):
        self.agent = QueryUnderstandingAgent(config={})
        self.config = {
             "agents": {
                "QueryUnderstandingAgent": {
                    "supported_commands": ["risk", "kb:", "updatekb", "create_agent_stub", "clear_scratchpad"]
                }
             }
        }

    @patch('core.utils.config_utils.load_config')
    async def test_execute_risk_query(self, mock_load_config):
        mock_load_config.return_value = self.config
        result = await self.agent.execute("risk ABC")
        self.assertEqual(result, ["DataRetrievalAgent"])

    @patch('core.utils.config_utils.load_config')
    async def test_execute_kb_query(self, mock_load_config):
        mock_load_config.return_value = self.config
        result = await self.agent.execute("kb:market_sentiment")
        self.assertEqual(result, ["DataRetrievalAgent"])

    @patch('core.utils.config_utils.load_config')
    async def test_execute_updatekb_query(self, mock_load_config):
        mock_load_config.return_value = self.config
        result = await self.agent.execute("updatekb key:value")
        self.assertEqual(result, [])

    @patch('core.utils.config_utils.load_config')
    async def test_execute_unknown_query(self, mock_load_config):
        mock_load_config.return_value = self.config
        with self.assertRaises(InvalidInputError) as context:
            await self.agent.execute("unknown command")
        self.assertEqual(context.exception.code, 103)  # Check for correct error code


    @patch('core.utils.config_utils.load_config')
    async def test_execute_empty_query(self, mock_load_config):
        mock_load_config.return_value = self.config
        with self.assertRaises(InvalidInputError) as context:
            await self.agent.execute("")
        self.assertEqual(context.exception.code, 103)

if __name__ == '__main__':
    unittest.main()
