import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any
from core.agents.fundamental_analyst_agent import FundamentalAnalystAgent
from core.schemas.agent_schema import AgentInput, AgentOutput

# Sample Data for Testing
MOCK_DATA = {
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
            "fcf_projection_years_total": 10,
            "initial_high_growth_period_years": 5,
            "initial_high_growth_rate": 0.10,
            "stable_growth_rate": 0.05,
            "discount_rate": 0.09,
            "terminal_growth_rate_perpetuity": 0.025
        },
        "market_data": {"share_price": 65.00, "shares_outstanding": 10000000}
    },
    "qualitative_company_info": {"management_assessment": "Experienced", "competitive_advantages": "Strong IP"},
    "industry_data_context": {"outlook": "Positive"},
    "economic_data_context": {"overall_outlook": "Stable"},
    "collateral_and_debt_details": {"loan_to_value_ratio": 0.6}
}

class MockResult:
    def __str__(self): return "AI Summary"

@pytest.fixture
def mock_kernel():
    kernel = MagicMock()
    # Mock skills.get_function
    kernel.skills.get_function.return_value = MagicMock()
    # Mock run_async
    async def mock_run_async(*args, **kwargs):
        return MockResult()
    kernel.run_async = AsyncMock(side_effect=mock_run_async)

    # Mock invoke (for v1)
    async def mock_invoke(*args, **kwargs):
        return MockResult()
    kernel.invoke = AsyncMock(side_effect=mock_invoke)

    return kernel

@pytest.fixture
def agent(mock_kernel):
    config = {
        "agent_id": "test_agent",
        "persona": "Test Analyst",
        "description": "Test"
    }
    # Mock Gemini Analyzer to avoid instantiation issues
    with patch("core.agents.fundamental_analyst_agent.GeminiFinancialReportAnalyzer") as MockGemini:
        mock_instance = MockGemini.return_value
        # Ensure analyze_report is awaitable and returns an object with model_dump
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"executive_summary": "Test Summary", "management_sentiment": "Positive"}
        mock_instance.analyze_report = AsyncMock(return_value=mock_result)

        agent = FundamentalAnalystAgent(config, kernel=mock_kernel)

    # Mock peer
    mock_peer = MagicMock()
    mock_peer.name = "DataRetrievalAgent"
    agent.add_peer_agent(mock_peer)

    # Mock Message Broker for send_message if needed (though we patch send_message directly)
    agent.message_broker = MagicMock()

    return agent

@pytest.mark.asyncio
async def test_legacy_execution_success(agent):
    # Mock send_message to return data
    async def mock_send_message(target, message, timeout=30.0):
        if target == "DataRetrievalAgent":
            return MOCK_DATA
        return None

    with patch.object(agent, 'send_message', side_effect=mock_send_message):
        result = await agent.execute("ABC_TEST")

        assert isinstance(result, dict)
        assert result["company_id"] == "ABC_TEST"
        assert result["enterprise_value"] is not None
        assert result["financial_health"] == "Strong"

@pytest.mark.asyncio
async def test_standard_execution_success(agent):
    async def mock_send_message(target, message, timeout=30.0):
        if target == "DataRetrievalAgent":
            return MOCK_DATA
        return None

    with patch.object(agent, 'send_message', side_effect=mock_send_message):
        input_obj = AgentInput(query="ABC_TEST")
        result = await agent.execute(input_obj)

        assert isinstance(result, AgentOutput)
        assert result.confidence > 0.8
        assert "Company Data for ABC_TEST" in result.sources
        assert result.metadata["company_id"] == "ABC_TEST"
        assert "Strong" in result.metadata["financial_health"]

@pytest.mark.asyncio
async def test_execution_failure_legacy(agent):
    with patch.object(agent, 'send_message', new_callable=AsyncMock) as mock_send:
        mock_send.return_value = None # Failure

        result = await agent.execute("FAIL_TEST")
        assert "error" in result
        assert "Could not retrieve data" in result["error"]

@pytest.mark.asyncio
async def test_execution_failure_standard(agent):
    with patch.object(agent, 'send_message', new_callable=AsyncMock) as mock_send:
        mock_send.return_value = None # Failure

        input_obj = AgentInput(query="FAIL_TEST")
        result = await agent.execute(input_obj)

        assert isinstance(result, AgentOutput)
        assert result.confidence == 0.0
        assert "Analysis failed" in result.answer
