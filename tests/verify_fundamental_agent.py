import asyncio
import logging
import json
from unittest.mock import MagicMock, patch
from core.agents.fundamental_analyst_agent import FundamentalAnalystAgent
from core.schemas.agent_schema import AgentInput, AgentOutput

# Setup logging
logging.basicConfig(level=logging.INFO)

# Mock Data
mock_data_package = {
    "company_info": {"name": "TestCompany Corp", "industry_sector": "Tech", "country": "USA"},
    "financial_data_detailed": {
        "income_statement": {"revenue": [1000, 1100, 1250], "net_income": [100, 120, 150], "ebitda": [150, 170, 200]},
        "balance_sheet": {"total_assets": [2000, 2100, 2200], "total_liabilities": [800, 850, 900],
                            "shareholders_equity": [1200, 1250, 1300], "cash_and_equivalents": [200, 250, 300],
                            "short_term_debt": [50, 50, 50], "long_term_debt": [500, 450, 400]},
        "cash_flow_statement": {"operating_cash_flow": [180, 200, 230], "investing_cash_flow": [-50, -60, -70],
                                "financing_cash_flow": [-30, -40, -50], "free_cash_flow": [130, 140, 160]},
        "key_ratios": {"debt_to_equity_ratio": 0.58, "net_profit_margin": 0.20, "current_ratio": 2.95, "interest_coverage_ratio": 13.6},
        "dcf_assumptions": {
            "fcf_projection_years_total": 5,
            "initial_high_growth_period_years": 3,
            "initial_high_growth_rate": 0.10,
            "stable_growth_rate": 0.05,
            "discount_rate": 0.09,
            "terminal_growth_rate": 0.025
        },
        "market_data": {"share_price": 65.00, "shares_outstanding": 10000000}
    },
    "qualitative_company_info": {"management_assessment": "Experienced"},
    "industry_data_context": {"outlook": "Positive"},
    "economic_data_context": {"overall_outlook": "Stable"},
    "collateral_and_debt_details": {"loan_to_value_ratio": 0.6}
}

async def mock_send_message(self, target_agent_name, message):
    if target_agent_name == 'DataRetrievalAgent' and message.get('data_type') == 'get_company_financials':
        return mock_data_package
    return None

async def run_verification():
    print("Initializing FundamentalAnalystAgent...")
    config = {"persona": "Financial Analyst", "description": "Test Agent"}
    agent = FundamentalAnalystAgent(config)

    # Mock the peer agent check
    agent.peer_agents = {'DataRetrievalAgent': MagicMock()}

    # Patch send_message
    with patch.object(FundamentalAnalystAgent, 'send_message', new=mock_send_message):
        print("Executing Analysis...")
        input_data = AgentInput(query="TEST_CORP")
        result = await agent.execute(input_data)

        print(f"Result Type: {type(result)}")
        if isinstance(result, AgentOutput):
            metadata = result.metadata
        else:
            metadata = result

        print("Checking for Scenario Analysis in Result...")
        dcf_scenarios = metadata.get("dcf_valuation_scenarios")

        if dcf_scenarios:
            print(f"DCF Scenarios Found: {json.dumps(dcf_scenarios, indent=2)}")
            assert "Base Case" in dcf_scenarios, "Base Case missing"
            assert "Bull Case" in dcf_scenarios, "Bull Case missing"
            assert "Bear Case" in dcf_scenarios, "Bear Case missing"
            print("SUCCESS: DCF Scenarios verified.")
        else:
            print("FAILURE: DCF Scenarios NOT found.")
            exit(1)

        summary = result.answer if isinstance(result, AgentOutput) else result.get("analysis_summary", "")
        print("\nChecking Summary for references...")
        if "Base Case" in summary and "Bull Case" in summary:
             print("SUCCESS: Summary contains scenario references.")
        else:
             print(f"FAILURE: Summary does not explicitly mention scenarios.\nSummary Preview: {summary[:200]}...")
             exit(1)

if __name__ == "__main__":
    asyncio.run(run_verification())
