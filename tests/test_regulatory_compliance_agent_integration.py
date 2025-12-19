import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

sys.path.append(os.getcwd())

# Mock heavy/external dependencies
sys.modules["nltk"] = MagicMock()
sys.modules["neo4j"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["bs4"] = MagicMock()

# Mock dependencies used in agent_base or regulatory_compliance_agent that we don't want to test
sys.modules["core.data_sources.political_landscape"] = MagicMock()

# We need to import RegulatoryComplianceAgent.
# Note: In the codebase, it might not yet inherit from AgentBase, so checking inheritance might fail initially.
try:
    from core.agents.regulatory_compliance_agent import RegulatoryComplianceAgent
    from core.agents.agent_base import AgentBase
except ImportError as e:
    print(f"Import failed: {e}")
    RegulatoryComplianceAgent = None

class TestRegulatoryComplianceIntegration(unittest.TestCase):
    def setUp(self):
        if RegulatoryComplianceAgent is None:
            self.skipTest("RegulatoryComplianceAgent could not be imported")

        self.config = {
            "agent_id": "reg_agent",
            "knowledge_graph_uri": "mock_uri",
            "llm_config": {}
        }

    def test_integration_logic(self):
        # We need to run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._test_integration_logic_async())
        finally:
            loop.close()

    async def _test_integration_logic_async(self):
        # We mock LLMPlugin specifically where it is imported in the agent module
        with patch("core.agents.regulatory_compliance_agent.LLMPlugin"), \
             patch("core.agents.regulatory_compliance_agent.PoliticalLandscapeLoader"), \
             patch.object(RegulatoryComplianceAgent, '_initialize_nlp_toolkit', return_value=MagicMock()), \
             patch.object(RegulatoryComplianceAgent, '_load_regulatory_knowledge', return_value={}), \
             patch.object(RegulatoryComplianceAgent, '_load_political_landscape', return_value={}):

            # Initialize agent
            agent = RegulatoryComplianceAgent(self.config)

            # Check inheritance (This verifies we made it an AgentBase)
            self.assertIsInstance(agent, AgentBase, "Agent should inherit from AgentBase")

            # Mock send_message from AgentBase
            # If send_message doesn't exist (because inheritance missing), this setting might work but assertIsInstance failed above
            agent.send_message = AsyncMock()

            # Mock internal methods to isolate integration logic
            agent._generate_compliance_report = MagicMock(return_value=("Report Content", []))
            agent._get_regulatory_updates = MagicMock(return_value=[("Source", "Title", "Summary")])
            agent._integrate_knowledge = MagicMock()
            agent._continuous_learning = MagicMock()
            agent._save_audit_trail = MagicMock()

            transactions = [{"id": "1", "amount": 100}]

            # Execute
            if not hasattr(agent, 'execute'):
                 self.fail("Agent does not have execute method")

            result = await agent.execute(transactions=transactions)

            # Assertions
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["compliance_report"], "Report Content")

            # Verify Integration with RiskAssessmentAgent
            agent.send_message.assert_called_with(
                "RiskAssessmentAgent",
                {
                    "source": "RegulatoryComplianceAgent",
                    "report": "Report Content",
                    "risk_type": "compliance",
                    "regulatory_updates": 1,
                    "violation_count": 0
                }
            )

            # Verify Audit Trail
            agent._save_audit_trail.assert_called()

            # Verify Knowledge Base Integration called
            agent._integrate_knowledge.assert_called()
            agent._continuous_learning.assert_called()

if __name__ == "__main__":
    unittest.main()
