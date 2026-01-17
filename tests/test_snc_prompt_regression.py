import unittest
import json
import sys
import os

# Ensure core is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents.snc_analyst_agent import SNCAnalystAgent

class TestSNCPromptRegression(unittest.TestCase):
    """
    Regression suite for SNC Analyst Agent prompt construction logic.
    Ensures that data is correctly formatted before being sent to the LLM/SK.
    """

    def setUp(self):
        # Config is irrelevant for these unit tests but required for init
        self.agent = SNCAnalystAgent({'peers': []})

    def test_financial_inputs_formatting(self):
        """Test formatting of financial data for prompt injection."""

        input_data = {
            "cash_flow_statement": {
                "free_cash_flow": [100, 120, 140],
                "cash_flow_from_operations": [200, 220, 240]
            },
            "key_ratios": {
                "debt_to_equity_ratio": 1.5,
                "current_ratio": 2.0
            },
            "market_data": {
                "annual_debt_service_placeholder": "50000",
                "payment_history_placeholder": "Current"
            },
            "dcf_assumptions": {
                "projected_fcf_placeholder": "150"
            }
        }

        expected_outputs = {
            "historical_fcf_str": "[100, 120, 140]",
            "historical_cfo_str": "[200, 220, 240]",
            "annual_debt_service_str": "50000",
            # ratios_summary_str should be a json string
            "projected_fcf_str": "150",
            "payment_history_status_str": "Current",
            "interest_capitalization_status_str": "No" # Default
        }

        result = self.agent._prepare_financial_inputs_for_sk(input_data)

        for key, value in expected_outputs.items():
            self.assertEqual(result.get(key), value, f"Mismatch for {key}")

        # Check JSON field specifically
        ratios_json = result.get("ratios_summary_str")
        self.assertTrue(isinstance(ratios_json, str))
        ratios = json.loads(ratios_json)
        self.assertEqual(ratios["debt_to_equity_ratio"], 1.5)

    def test_missing_data_handling(self):
        """Ensure robust handling of missing data in prompt construction."""
        input_data = {} # Empty

        result = self.agent._prepare_financial_inputs_for_sk(input_data)

        self.assertEqual(result["historical_fcf_str"], "['N/A']")
        self.assertEqual(result["annual_debt_service_str"], "Not Available")
        self.assertEqual(result["ratios_summary_str"], "Not available")

    def test_qualitative_inputs_formatting(self):
        """Test formatting of qualitative data."""
        input_data = {
            "revenue_cashflow_stability_notes_placeholder": "Very stable.",
            "financial_deterioration_notes_placeholder": "None."
        }

        result = self.agent._prepare_qualitative_inputs_for_sk(input_data)

        self.assertEqual(result["qualitative_notes_stability_str"], "Very stable.")
        self.assertEqual(result["notes_financial_deterioration_str"], "None.")

if __name__ == '__main__':
    unittest.main()
