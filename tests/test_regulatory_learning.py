import os
import sys
import unittest
from unittest.mock import mock_open, patch

# Adjust path to import core
sys.path.append(os.getcwd())

from core.agents.regulatory_compliance_agent import RegulatoryComplianceAgent


class TestRegulatoryComplianceAgentLearning(unittest.TestCase):
    def setUp(self):
        self.config = {
            "agent_id": "test_compliance_agent",
            "llm_config": {"model": "mock_model"},
            "knowledge_graph_uri": "bolt://localhost:7687",
            "regulatory_api_key": "mock_key"
        }

        # Mock dependencies
        self.mock_llm_plugin = patch('core.agents.regulatory_compliance_agent.LLMPlugin').start()
        self.mock_nltk = patch('core.agents.regulatory_compliance_agent.nltk').start()
        self.mock_requests = patch('core.agents.regulatory_compliance_agent.requests').start()
        self.mock_graph_db = patch('core.agents.regulatory_compliance_agent.GraphDatabase').start()
        self.mock_loader = patch('core.agents.regulatory_compliance_agent.PoliticalLandscapeLoader').start()

        # Setup mock return values
        self.mock_loader.return_value.load_landscape.return_value = {"US": {"president": "Mock", "recent_developments": []}}

        # Instantiate agent
        # We mock open inside load_learned_profile to start with clean state
        with patch("builtins.open", mock_open(read_data='{}')) as mock_file:
            # We need to make sure os.path.exists returns False so it uses defaults
            with patch("os.path.exists", return_value=False):
                self.agent = RegulatoryComplianceAgent(self.config)

    def tearDown(self):
        patch.stopall()
        # Clean up any created files
        if os.path.exists("data/compliance_learned_profile.json"):
            os.remove("data/compliance_learned_profile.json")

    def test_initial_analysis(self):
        transaction = {"id": "t1", "amount": 15000, "customer": "CustA"}
        result = self.agent._analyze_transaction(transaction)
        # Should detect threshold violation
        self.assertIn("Transaction amount exceeds threshold", result["violated_rules"])
        # Expect default risk score (currently 0.5)
        self.assertEqual(result["risk_score"], 0.5)

    def test_learning_mechanism(self):
        # 1. Setup a scenario
        transaction = {"id": "t2", "amount": 5000, "customer": "HighRiskCust", "country": "US"}

        # 2. Simulate learning: This customer has violated rules before
        past_results = [{
            "transaction_id": "past_t",
            "entity_id": "HighRiskCust",
            "violated_rules": ["Some Rule"]
        }]

        # Mock saving to file to avoid actual I/O in test logic if possible,
        # but _continuous_learning calls _save_learned_profile which calls open.
        # We can just let it write to the temp file since we clean it up.

        self.agent._continuous_learning(past_results)

        # Now analyze current transaction
        result = self.agent._analyze_transaction(transaction)

        # Should have a risk score > 0 due to entity modifier
        # Default modifier increment is 0.05 * 0.1 = 0.005. Wait.
        # Code: modifiers[entity_id] = modifiers.get(entity_id, 0.0) + (learning_rate * 0.1)
        # 0.05 * 0.1 = 0.005.
        # So risk score should be 0.005.

        self.assertGreater(result["risk_score"], 0.0)
        self.assertAlmostEqual(result["risk_score"], 0.005)

    def test_feedback_mechanism(self):
        # Test adjusting rule weights
        rule_name = "threshold_violation"
        initial_weight = self.agent.learned_params["rule_weights"][rule_name]

        # Provide feedback that this rule was a False Positive (incorrect assessment)
        feedback = {
            "rule_id": rule_name,
            "correct_assessment": False,
            "transaction_id": "t_fp"
        }
        self.agent.process_feedback(feedback)

        new_weight = self.agent.learned_params["rule_weights"][rule_name]
        self.assertLess(new_weight, initial_weight)

if __name__ == '__main__':
    unittest.main()
