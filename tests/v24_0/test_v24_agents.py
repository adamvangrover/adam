import unittest
from unittest.mock import MagicMock, AsyncMock
import asyncio
from core.agents.meta_agents.evolutionary_architect import EvolutionaryArchitect
from core.agents.meta_agents.didactic_architect import DidacticArchitect
from core.agents.meta_agents.chronos_agent import Chronos
from core.llm.base_llm_engine import BaseLLMEngine

class TestV24Agents(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mock_llm = MagicMock(spec=BaseLLMEngine)
        # Mock generate_response to be an async method returning a string
        self.mock_llm.generate_response = AsyncMock(return_value="Mocked LLM Response")

    async def test_evolutionary_architect_init(self):
        agent = EvolutionaryArchitect(llm_engine=self.mock_llm)
        self.assertEqual(agent.name, "EvolutionaryArchitect")
        self.assertIn("Evolution", agent.role)

    async def test_evolutionary_architect_run(self):
        agent = EvolutionaryArchitect(llm_engine=self.mock_llm)
        context = {
            "user_query": "Optimize the pricing engine",
            "architecture_summary": "Current implementation uses brute force."
        }
        # execute is async
        result = await agent.execute(context)
        self.assertIn("response", result)
        self.assertIn("action_plan", result)
        self.assertEqual(result["response"], "Mocked LLM Response")
        self.assertEqual(len(agent.evolution_log), 1)

    async def test_didactic_architect_init(self):
        agent = DidacticArchitect(llm_engine=self.mock_llm)
        self.assertEqual(agent.name, "DidacticArchitect")

    async def test_didactic_architect_run(self):
        agent = DidacticArchitect(llm_engine=self.mock_llm)
        context = {
            "target_content": "def risky_function(): pass",
            "type": "tutorial"
        }
        result = await agent.execute(context)
        self.assertIn("response", result)
        self.assertEqual(result["artifact_type"], "tutorial")

    async def test_chronos_init(self):
        agent = Chronos(llm_engine=self.mock_llm)
        self.assertEqual(agent.name, "Chronos")

    async def test_chronos_run(self):
        agent = Chronos(llm_engine=self.mock_llm)
        context = {
            "user_query": "What happened during the 2008 crash?",
            "market_snapshot": {"sp500": 4000}
        }
        result = await agent.execute(context)
        self.assertIn("temporal_context", result)
        self.assertIn("strategy", result["temporal_context"])
        # The mock LLM returns "Mocked LLM Response", but the code does .strip().lower()
        self.assertEqual(result["temporal_context"]["strategy"], "mocked llm response")

if __name__ == '__main__':
    unittest.main()
