import pytest
import asyncio
from core.v30_architecture.python_intelligence.agents.regulatory_compliance_agent import RegulatoryComplianceAgent

@pytest.mark.asyncio
async def test_compliance_agent_execute():
    agent = RegulatoryComplianceAgent()

    # Test standard trade
    result = await agent.execute(target="portfolio", action="standard_trade")
    assert result["target"] == "portfolio"
    assert result["action"] == "standard_trade"
    assert result["is_compliant"] == True
    assert len(result["warnings"]) == 0
    assert result["status"] == "COMPLETED"

    # Test high risk trade
    result = await agent.execute(target="crypto", action="high_risk_trade")
    assert result["target"] == "crypto"
    assert result["action"] == "high_risk_trade"
    assert result["is_compliant"] == False
    assert len(result["warnings"]) == 1
    assert "manual approval" in result["warnings"][0]
