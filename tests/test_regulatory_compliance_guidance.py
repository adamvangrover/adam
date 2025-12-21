import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Set python path
sys.path.append(os.getcwd())

# Mock external dependencies BEFORE importing the module
sys.modules["nltk"] = MagicMock()
sys.modules["nltk.stem"] = MagicMock()
sys.modules["neo4j"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["bs4"] = MagicMock()
sys.modules["core.llm_plugin"] = MagicMock()

# Now import the class
from core.agents.regulatory_compliance_agent import RegulatoryComplianceAgent


class TestRegulatoryComplianceAgent(unittest.TestCase):
    def setUp(self):
        self.config = {"knowledge_graph_uri": "mock_uri"}

    def test_provide_guidance_calls_llm(self):
        # We need to mock methods called in __init__
        with patch.object(RegulatoryComplianceAgent, '_initialize_nlp_toolkit'), \
             patch.object(RegulatoryComplianceAgent, '_load_regulatory_knowledge', return_value={"Rule1": "Must be compliant"}), \
             patch.object(RegulatoryComplianceAgent, '_load_political_landscape', return_value={"Politics": "Stable"}):

            agent = RegulatoryComplianceAgent(self.config)

            # Mock the LLM instance on the agent
            mock_llm = MagicMock()
            mock_llm.generate_text.return_value = "This is LLM guidance."
            agent.llm = mock_llm

            question = "How to comply?"
            guidance = agent.provide_guidance(question)

            self.assertEqual(guidance, "This is LLM guidance.")
            mock_llm.generate_text.assert_called_once()

            # Verify prompt contains context
            args, kwargs = mock_llm.generate_text.call_args
            prompt = args[0]
            self.assertIn("How to comply?", prompt)
            self.assertIn("Rule1", prompt)
            self.assertIn("Politics", prompt)

if __name__ == '__main__':
    unittest.main()
