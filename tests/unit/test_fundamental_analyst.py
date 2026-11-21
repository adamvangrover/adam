import pytest
import asyncio
from unittest.mock import MagicMock, patch
from core.agents.fundamental_analyst_agent import FundamentalAnalystAgent
from core.schemas.agent_schema import AgentInput, AgentOutput

# Mock dependencies
class MockMemoryManager:
    def query_history(self, *args, **kwargs):
        return []
    def search_similar(self, *args, **kwargs):
        return []
    def save_analysis(self, *args, **kwargs):
        pass

class MockGeminiAnalyzer:
    async def analyze_report(self, *args, **kwargs):
        return MagicMock(model_dump=lambda: {"executive_summary": "Good", "management_sentiment": "Positive"})

@pytest.fixture
def agent():
    config = {
        "persona": "Test Analyst",
        "description": "Test",
        "peers": ["DataRetrievalAgent"]
    }
    with patch("core.agents.fundamental_analyst_agent.VectorMemoryManager", return_value=MockMemoryManager()), \
         patch("core.agents.fundamental_analyst_agent.GeminiFinancialReportAnalyzer", return_value=MockGeminiAnalyzer()):
        agent = FundamentalAnalystAgent(config)
        agent.peer_agents = {"DataRetrievalAgent": MagicMock()} # Mock peer
        return agent

@pytest.mark.asyncio
async def test_execute_standard_input(agent):
    # Mock retrieve_company_data
    mock_data = {
        "company_info": {"name": "Test Corp"},
        "financial_data_detailed": {
            "income_statement": {"revenue": [100, 110], "net_income": [10, 11], "ebitda": [20, 22]},
            "balance_sheet": {"total_assets": [200], "total_liabilities": [100], "shareholders_equity": [100], "cash_and_equivalents": [50], "short_term_debt": [10], "long_term_debt": [40]},
            "cash_flow_statement": {"free_cash_flow": [10, 12, 15]},
            "market_data": {"share_price": 50, "shares_outstanding": 10},
            "dcf_assumptions": {"discount_rate": 0.1, "terminal_growth_rate": 0.02}
        },
        "qualitative_company_info": {"info": "some text"}
    }

    with patch.object(agent, 'retrieve_company_data', return_value=mock_data):
        input_data = AgentInput(query="TEST", context={})
        result = await agent.execute(input_data)

        assert isinstance(result, AgentOutput)
        assert result.confidence > 0.0
        # Check if the company_id (TEST) is present in the metadata
        assert "TEST" in str(result.metadata)
        # Check if insights are present
        assert "gemini_insights" in result.metadata

@pytest.mark.asyncio
async def test_execute_failure(agent):
    with patch.object(agent, 'retrieve_company_data', return_value=None):
        input_data = AgentInput(query="FAIL", context={})
        result = await agent.execute(input_data)

        assert isinstance(result, AgentOutput)
        assert result.confidence == 0.0
        assert "Analysis failed" in result.answer
