import sys
import os
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestCodeAlchemist(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {
            "openai_api_key": "test_key",
            "validation_tool_url": "http://test-url",
        }

    @patch('core.agents.code_alchemist.LLMPlugin')
    async def test_initialization(self, MockLLMPlugin):
        from core.agents.code_alchemist import CodeAlchemist
        agent = CodeAlchemist(self.config)
        self.assertEqual(agent.validation_tool_url, "http://test-url")
        # Check defaults
        self.assertIn("performance", agent.optimization_strategies)

    @patch('core.agents.code_alchemist.LLMPlugin')
    async def test_load_system_prompt(self, MockLLMPlugin):
        from core.agents.code_alchemist import CodeAlchemist
        agent = CodeAlchemist(self.config)
        prompt = agent._load_system_prompt()
        if prompt:
            self.assertIn("Code Alchemist", prompt)
            self.assertIn("LIB-META-008", prompt)
        else:
            print("Warning: System prompt not found during test.")

    @patch('core.agents.code_alchemist.LLMPlugin')
    async def test_construct_generation_prompt(self, MockLLMPlugin):
        from core.agents.code_alchemist import CodeAlchemist
        agent = CodeAlchemist(self.config)
        # Mock system prompt template if not loaded (for environment consistency)
        if not agent.system_prompt_template:
             agent.system_prompt_template = "Intent: {{intent}} Context: {{context}} Constraints: {{constraints}} Knowledge: {{relevant_knowledge}}"

        prompt = agent.construct_generation_prompt(
            intent="Test Intent",
            context={"env": "test"},
            constraints={"fast": True},
            relevant_knowledge="Know stuff"
        )
        self.assertIn("Test Intent", prompt)

        # Verify JSON dumping if template is loaded
        if "LIB-META-008" in (agent.system_prompt_template or ""):
             self.assertIn('"fast": true', prompt) # Check JSON representation

    @patch('core.agents.code_alchemist.LLMPlugin')
    async def test_validate_code_semantics(self, MockLLMPlugin):
        from core.agents.code_alchemist import CodeAlchemist

        # Setup mock to return semantic analysis JSON
        mock_llm_instance = MockLLMPlugin.return_value
        mock_llm_instance.get_completion = AsyncMock(return_value='{"semantic_errors": null}')

        agent = CodeAlchemist(self.config)

        # Test valid code syntax
        code = "def foo(): return 1"
        result = await agent.validate_code(code)

        self.assertIsNone(result["syntax_errors"])
        self.assertIsNone(result["semantic_errors"])

    @patch('core.agents.code_alchemist.LLMPlugin')
    async def test_extract_json_resilience(self, MockLLMPlugin):
        from core.agents.code_alchemist import CodeAlchemist
        agent = CodeAlchemist(self.config)

        # Test clean JSON
        res1 = agent._extract_json('{"key": "value"}')
        self.assertEqual(res1, {"key": "value"})

        # Test JSON in markdown
        res2 = agent._extract_json('Here is json: ```json\n{"key": "value"}\n```')
        self.assertEqual(res2, {"key": "value"})

        # Test JSON with extra text
        res3 = agent._extract_json('Some text {"key": "value"} end text')
        self.assertEqual(res3, {"key": "value"})

        # Test invalid JSON
        res4 = agent._extract_json('No json here')
        self.assertEqual(res4, {})

if __name__ == '__main__':
    unittest.main()
