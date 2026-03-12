import pytest
from unittest.mock import MagicMock, AsyncMock
from core.agents.specialized.distressed_debt_agent import DistressedDebtAgent, DistressedDebtAnalysis, DebtTranche

@pytest.fixture
def distressed_agent():
    config = {"name": "DistressedDebtAgent", "provider": "mock"}
    agent = DistressedDebtAgent(config)
    agent.llm_plugin = MagicMock()
    return agent

@pytest.mark.asyncio
async def test_distressed_debt_agent_execution_success(distressed_agent):
    # Mock successful LLM output
    mock_response = DistressedDebtAnalysis(
        issuer_name="Zombie Corp",
        enterprise_value_distressed=400.0,
        implied_default_probability=0.25,
        restructuring_strategy="Equitization",
        tranches=[
            DebtTranche(
                facility_name="Term Loan B",
                amount_outstanding=300.0,
                lien_position=1,
                recovery_estimate_pct=0.75,
                covenant_breach_probability=0.3
            )
        ]
    )
    distressed_agent.llm_plugin.generate_structured.return_value = (mock_response, None)

    result = await distressed_agent.execute(
        issuer_name="Zombie Corp",
        financials={"ebitda": 50},
        debt_structure=[{"facility_name": "Term Loan B", "amount": 300, "lien": 1}]
    )

    assert result["status"] == "success"
    assert result["issuer"] == "Zombie Corp"

    analysis = result["distressed_analysis"]
    assert analysis["enterprise_value_distressed"] == 400.0
    assert analysis["implied_default_probability"] == 0.25
    assert analysis["restructuring_strategy"] == "Equitization"
    assert len(analysis["tranches"]) == 1
    assert analysis["tranches"][0]["facility_name"] == "Term Loan B"

@pytest.mark.asyncio
async def test_distressed_debt_agent_execution_fallback(distressed_agent):
    # Mock LLM failure (e.g., returned None)
    distressed_agent.llm_plugin.generate_structured.return_value = (None, None)

    result = await distressed_agent.execute(
        issuer_name="Failing Co",
        financials={"ebitda": -10},
        debt_structure=[
            {"facility_name": "1L Term Loan", "amount": 200, "lien": 1},
            {"facility_name": "2L Notes", "amount": 100, "lien": 2}
        ]
    )

    assert result["status"] == "success"
    assert result["issuer"] == "Failing Co"

    analysis = result["distressed_analysis"]
    assert len(analysis["tranches"]) == 2

    # Check fallback recovery logic
    assert analysis["tranches"][0]["recovery_estimate_pct"] == 0.80
    assert analysis["tranches"][1]["recovery_estimate_pct"] == 0.40
    assert analysis["enterprise_value_distressed"] == 240.0  # (200 + 100) * 0.8
