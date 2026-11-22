import asyncio

from core.agents.templates.v30_template_agent import TemplateAgentV30
from core.credit_sentinel.covenant_agent import CovenantAnalystAgent
from core.schemas.agent_schema import AgentInput, AgentOutput


class _FakeMCP:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return self.result


def _run(agent, context):
    return asyncio.run(agent.execute(AgentInput(query="covenant check", context=context)))


def test_covenant_agent_is_v30_and_emits_provenance():
    mcp = _FakeMCP({"fccr": 3.1, "headroom_bps": 1100, "is_breached": False})
    agent = CovenantAnalystAgent(mcp)
    assert isinstance(agent, TemplateAgentV30)
    out = _run(agent, {"financial_statements": {"EBITDA": 310.0, "interest_expense": 100.0}})
    assert isinstance(out, AgentOutput)
    assert out.metadata["next_step"] == "update_hdkg"
    assert out.metadata["confidence_tier"] == "autonomous"
    assert out.provenance_trace
    assert mcp.calls[0][0] == "calculate_fccr_headroom"


def test_covenant_agent_breach_and_missing_data():
    breach = CovenantAnalystAgent(_FakeMCP({"fccr": 1.4, "headroom_bps": -600, "is_breached": True}))
    out = _run(breach, {"financial_statements": {"EBITDA": 140.0, "interest_expense": 100.0}})
    assert out.metadata["next_step"] == "consult_legal"
    missing = _run(CovenantAnalystAgent(_FakeMCP({})), {})
    assert missing.confidence == 0.1
    assert missing.metadata["confidence_tier"] == "hard_abort"
