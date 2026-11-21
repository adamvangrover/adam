import unittest
import asyncio
import json
import os
import shutil
import sys
from unittest.mock import MagicMock, patch

sys.path.append('.')

from core.agents.snc_analyst_agent import SNCAnalystAgent, SNCRating

class TestSNCCompliance(unittest.TestCase):
    def setUp(self):
        self.config = {
            'persona': "Test Agent",
            'comptrollers_handbook_SNC': {},
            'occ_guidelines_SNC': {},
            'peers': ['DataRetrievalAgent']
        }
        self.mock_kernel = MagicMock()
        # Mocking SK to be missing or failing
        self.mock_kernel.skills.get_function.return_value = None

        self.defense_file = "test_defense_file.json"

    def tearDown(self):
        if os.path.exists(self.defense_file):
            os.remove(self.defense_file)

    async def async_test_compliance_failure(self):
        agent = SNCAnalystAgent(self.config, self.mock_kernel)
        agent.peer_agents['DataRetrievalAgent'] = MagicMock()

        # Mock send_message for Data Retrieval
        async def mock_send_message(target, message):
            return {
                "company_info": {"name": "Risky Corp"},
                "financial_data_detailed": {
                    "key_ratios": {
                        "debt_to_equity_ratio": 4.5, # Fail (> 3.0)
                        "net_profit_margin": 0.1,
                        "current_ratio": 2.0,
                        "interest_coverage_ratio": 3.0
                    },
                    "cash_flow_statement": {},
                    "market_data": {},
                    "dcf_assumptions": {}
                },
                "qualitative_company_info": {},
                "industry_data_context": {},
                "economic_data_context": {},
                "collateral_and_debt_details": {}
            }

        agent.send_message = mock_send_message

        # Force fallback by making SK return junk
        agent.run_semantic_kernel_skill = MagicMock(return_value=asyncio.Future())
        agent.run_semantic_kernel_skill.return_value.set_result("Confused AI")

        rating, rationale = await agent.execute(company_id="RISKY_CORP")

        self.assertEqual(rating, SNCRating.SUBSTANDARD)
        self.assertIn("Compliance Validation Failed", rationale)
        self.assertIn("Leverage Breach", rationale)

        # Generate defense file
        agent.generate_defense_file(self.defense_file)
        self.assertTrue(os.path.exists(self.defense_file))

        with open(self.defense_file, 'r') as f:
            log = json.load(f)
            compliance_events = [e for e in log if e['event_type'] == "COMPLIANCE_CHECK"]
            self.assertTrue(len(compliance_events) > 0)
            self.assertFalse(compliance_events[0]['details']['passed'])

    def test_compliance_failure(self):
        asyncio.run(self.async_test_compliance_failure())

    async def async_test_compliance_pass(self):
        agent = SNCAnalystAgent(self.config, self.mock_kernel)
        agent.peer_agents['DataRetrievalAgent'] = MagicMock()

        async def mock_send_message(target, message):
            return {
                "company_info": {"name": "Safe Corp"},
                "financial_data_detailed": {
                    "key_ratios": {
                        "debt_to_equity_ratio": 1.0, # Pass
                        "net_profit_margin": 0.1,
                        "current_ratio": 2.0,
                        "interest_coverage_ratio": 3.0
                    },
                    "cash_flow_statement": {},
                    "market_data": {},
                    "dcf_assumptions": {}
                },
                "qualitative_company_info": {},
                "industry_data_context": {},
                "economic_data_context": {},
                "collateral_and_debt_details": {}
            }

        agent.send_message = mock_send_message

        # Force fallback to test compliance pass -> legacy fallback pass
        agent.run_semantic_kernel_skill = MagicMock(return_value=asyncio.Future())
        agent.run_semantic_kernel_skill.return_value.set_result("Confused AI")

        rating, rationale = await agent.execute(company_id="SAFE_CORP")

        self.assertEqual(rating, SNCRating.PASS)

        agent.generate_defense_file(self.defense_file)
        with open(self.defense_file, 'r') as f:
            log = json.load(f)
            compliance_events = [e for e in log if e['event_type'] == "COMPLIANCE_CHECK"]
            self.assertTrue(compliance_events[0]['details']['passed'])

    def test_compliance_pass(self):
        asyncio.run(self.async_test_compliance_pass())

    async def async_test_compliance_override_primary(self):
        """
        Test that compliance validators override the Agent/SK if it hallucinates a PASS
        despite bad financials.
        """
        agent = SNCAnalystAgent(self.config, self.mock_kernel)
        agent.peer_agents['DataRetrievalAgent'] = MagicMock()

        async def mock_send_message(target, message):
            return {
                "company_info": {"name": "Hallucination Corp"},
                "financial_data_detailed": {
                    "key_ratios": {
                        "debt_to_equity_ratio": 5.0, # CRITICAL FAIL
                        "net_profit_margin": 0.1,
                        "current_ratio": 2.0,
                        "interest_coverage_ratio": 3.0
                    },
                    "cash_flow_statement": {},
                    "market_data": {},
                    "dcf_assumptions": {}
                },
                "qualitative_company_info": {},
                "industry_data_context": {},
                "economic_data_context": {},
                "collateral_and_debt_details": {}
            }

        agent.send_message = mock_send_message

        # Mock SK to return PASS (Hallucination)
        # We need to mock run_semantic_kernel_skill to return strings that parse to PASS
        async def mock_run_sk_pass(collection, skill, inputs):
            if skill == "CollateralRiskAssessment":
                return "Assessment: Pass\nJustification: Good collateral."
            if skill == "AssessRepaymentCapacity":
                return "Assessment: Strong\nJustification: Good repayment."
            if skill == "AssessNonAccrualStatusIndication":
                return "Assessment: Accrual Appropriate\nJustification: Good."
            return ""

        agent.run_semantic_kernel_skill = mock_run_sk_pass

        # Explicitly ensure kernel and skills are present
        agent.kernel = MagicMock()
        agent.kernel.skills = MagicMock() # Ensure hasattr(kernel, 'skills') is True

        # Assign method to instance
        agent.run_semantic_kernel_skill = mock_run_sk_pass

        rating, rationale = await agent.execute(company_id="HALLUCINATION_CORP")

        # Should be downgraded to SUBSTANDARD
        self.assertEqual(rating, SNCRating.SUBSTANDARD)
        self.assertIn("CRITICAL COMPLIANCE VIOLATION", rationale)
        self.assertIn("Leverage Breach", rationale)

        # Verify Audit Log
        agent.generate_defense_file(self.defense_file)
        with open(self.defense_file, 'r') as f:
            log = json.load(f)
            # Check for COMPLIANCE_CHECK_PRIMARY event
            events = [e for e in log if e['event_type'] == "COMPLIANCE_CHECK_PRIMARY"]
            self.assertTrue(len(events) > 0)
            self.assertFalse(events[0]['details']['passed'])
            self.assertEqual(events[0]['details']['original_rating'], "Pass")

    def test_compliance_override_primary(self):
        asyncio.run(self.async_test_compliance_override_primary())

if __name__ == '__main__':
    unittest.main()
