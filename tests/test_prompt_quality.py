import pytest
import asyncio
from unittest.mock import MagicMock, patch
import json
from typing import Dict, Any

from core.agents.snc_analyst_agent import SNCAnalystAgent, SNCRating

# Mock Data
MOCK_CONFIG = {
    'persona': "Test Analyst",
    'comptrollers_handbook_SNC': {"version": "2024.Q1"},
    'occ_guidelines_SNC': {"version": "2024-03"},
    'peers': ['DataRetrievalAgent']
}

# Golden Set: Input Scenarios and Expected Outcomes
GOLDEN_SET = [
    {
        "id": "SCENARIO_1_HEALTHY",
        "input": {
            "company_id": "HEALTHY_CORP",
            "financials": {
                "key_ratios": {"debt_to_equity_ratio": 1.0, "net_profit_margin": 0.2, "current_ratio": 2.0, "interest_coverage_ratio": 5.0, "tier_1_capital_ratio": 0.12},
                "cash_flow_statement": {"free_cash_flow": [100], "cash_flow_from_operations": [150]},
                "market_data": {}
            }
        },
        "expected_rating": SNCRating.PASS,
        "required_phrases": ["Strong repayment", "Pass-rated collateral"]
    },
    {
        "id": "SCENARIO_2_LEVERAGE_BREACH",
        "input": {
            "company_id": "LEVERAGED_CORP",
            "financials": {
                "key_ratios": {"debt_to_equity_ratio": 4.5, "net_profit_margin": 0.05, "current_ratio": 1.2, "interest_coverage_ratio": 2.0, "tier_1_capital_ratio": 0.08},
                "cash_flow_statement": {"free_cash_flow": [10], "cash_flow_from_operations": [50]},
                "market_data": {}
            }
        },
        "expected_rating": SNCRating.SUBSTANDARD, # Should trigger Compliance Breach
        "required_phrases": ["CRITICAL COMPLIANCE VIOLATION", "Leverage Breach"]
    }
]

@pytest.mark.asyncio
async def test_snc_prompt_regression():
    """
    CI/CP: Prompt Regression Test.
    Runs agent against Golden Set and validates outputs using a deterministic Judge.
    """
    print("\n--- Starting Prompt Regression Suite ---")

    for case in GOLDEN_SET:
        print(f"Running Case: {case['id']}")

        # Setup Agent
        agent = SNCAnalystAgent(config=MOCK_CONFIG)
        agent.peer_agents['DataRetrievalAgent'] = MagicMock()

        # Enable SK path by setting kernel properties
        agent.kernel = MagicMock()
        agent.kernel.skills = MagicMock()

        # Mock A2A Response
        mock_data_package = {
            "company_info": {"name": case['input']['company_id']},
            "financial_data_detailed": case['input']['financials'],
            "qualitative_company_info": {"management_assessment": "Average"},
            "industry_data_context": {"sector": "General"},
            "economic_data_context": {"vix": 20.0},
            "collateral_and_debt_details": {"loan_to_value_ratio": 0.5}
        }

        async def mock_send_message(*args, **kwargs):
            return mock_data_package

        class MockKernel:
             async def run_semantic_kernel_skill(self, *args, **kwargs):
                 # Debug prints
                 print(f"DEBUG: Mock SK Called with args: {args}")
                 if len(args) > 1:
                     skill_name = args[1]
                     if "Collateral" in skill_name: return "Assessment: Pass\nJustification: Good."
                     if "Repayment" in skill_name: return "Assessment: Strong\nJustification: Good."
                     if "NonAccrual" in skill_name: return "Assessment: Accrual Appropriate\nJustification: Good."
                 return "Assessment: Unknown\nJustification: None"

        mock_kernel_instance = MockKernel()

        with patch.object(agent, 'send_message', side_effect=mock_send_message):
             # Force replacement of method
             agent.run_semantic_kernel_skill = mock_kernel_instance.run_semantic_kernel_skill

             rating, rationale = await agent.execute(company_id=case['input']['company_id'])

             # Judge
             print(f"  Result: {rating.value if rating else 'None'}")
             # print(f"  Rationale: {rationale[:100]}...")

             # Check Rating
             assert rating == case['expected_rating'], f"Rating Mismatch for {case['id']}: Expected {case['expected_rating']}, got {rating}"

             # Check Phrases
             for phrase in case['required_phrases']:
                 assert phrase in rationale, f"Missing phrase '{phrase}' in rationale for {case['id']}. Actual: {rationale[:200]}..."

        print(f"Case {case['id']} Passed.")
