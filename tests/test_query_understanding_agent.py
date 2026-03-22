# tests/test_query_understanding_agent.py

import unittest
import pytest
from unittest.mock import patch
from core.agents.query_understanding_agent import QueryUnderstandingAgent
from core.system.error_handler import InvalidInputError


class TestQueryUnderstandingAgent(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # We patch LLMPlugin inside the test methods instead to avoid setUp patch issues
        self.patcher = patch('core.agents.query_understanding_agent.LLMPlugin')
        self.mock_llm_plugin_cls = self.patcher.start()
        self.mock_llm_instance = self.mock_llm_plugin_cls.return_value

        self.agent = QueryUnderstandingAgent(config={})
        self.config = {
            "agents": {
                "QueryUnderstandingAgent": {
                    "supported_commands": ["risk", "kb:", "updatekb", "create_agent_stub", "clear_scratchpad"]
                }
            }
        }

    def tearDown(self):
        self.patcher.stop()

    @pytest.mark.asyncio
    @patch('core.utils.config_utils.load_config')
    async def test_execute_risk_query(self, mock_load_config):
        mock_load_config.return_value = self.config
        self.agent.llm_plugin.generate_text.return_value = '{"agents": ["DataRetrievalAgent"]}'
        result = await self.agent.execute("risk ABC")
        self.assertEqual(result, ["DataRetrievalAgent"])

    @pytest.mark.asyncio
    @patch('core.utils.config_utils.load_config')
    async def test_execute_kb_query(self, mock_load_config):
        mock_load_config.return_value = self.config
        self.agent.llm_plugin.generate_text.return_value = '{"agents": ["DataRetrievalAgent"]}'
        result = await self.agent.execute("kb:market_sentiment")
        self.assertEqual(result, ["DataRetrievalAgent"])

    @pytest.mark.asyncio
    @patch('core.utils.config_utils.load_config')
    async def test_execute_updatekb_query(self, mock_load_config):
        mock_load_config.return_value = self.config
        self.agent.llm_plugin.generate_text.return_value = '{"agents": []}'
        result = await self.agent.execute("updatekb key:value")
        self.assertEqual(result, [])

    @pytest.mark.asyncio
    @patch('core.utils.config_utils.load_config')
    async def test_execute_unknown_query(self, mock_load_config):
        mock_load_config.return_value = self.config
        self.agent.llm_plugin.generate_text.return_value = '{"agents": []}'
        result = await self.agent.execute("hello world")
        self.assertEqual(result, [])

    @pytest.mark.asyncio
    @patch('core.utils.config_utils.load_config')
    async def test_execute_empty_query(self, mock_load_config):
        mock_load_config.return_value = self.config
        self.agent.llm_plugin.generate_text.return_value = '{"agents": []}'
        result = await self.agent.execute("")
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()
