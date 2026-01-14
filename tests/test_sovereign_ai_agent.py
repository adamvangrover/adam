
import unittest
import asyncio
from core.agents.specialized.sovereign_ai_analyst_agent import SovereignAIAnalystAgent

class TestSovereignAIAnalystAgent(unittest.TestCase):
    def setUp(self):
        self.config = {
            'persona': "Test Analyst",
            'description': "Test Description",
            'expertise': ["Test Expertise"]
        }
        self.agent = SovereignAIAnalystAgent(self.config)

    def test_initialization(self):
        self.assertEqual(self.agent.persona, "Test Analyst")
        self.assertEqual(self.agent.expertise, ["Test Expertise"])

    def test_execute(self):
        async def run_execute():
            return await self.agent.execute()

        result = asyncio.run(run_execute())
        self.assertIn("agent", result)
        self.assertIn("analysis", result)
        self.assertIn("data_context", result)
        self.assertEqual(result["agent"], "Test Analyst")
        self.assertIn("138.5", result["analysis"]) # Check for GPR index in narrative

if __name__ == '__main__':
    unittest.main()
