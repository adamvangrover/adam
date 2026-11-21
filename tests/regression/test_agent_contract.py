"""Seam tests for the unified agent contract (core/schemas/agent_schema.py + AgentBase wrapping)."""
import asyncio

import pytest

from core.agents.agent_base import AgentBase
from core.agents.pydantic_agent_base import PydanticAgentBase
from core.agents.templates.v30_template_agent import TemplateAgentV30
from core.schemas.agent_schema import AgentInput, AgentOutput
from src.pdil.models import build_provenance_header


class _EchoAgent(AgentBase):
    calls = 0

    async def execute(self, *args, **kwargs):
        _EchoAgent.calls += 1
        return {"echo": args, "kw": kwargs}


class _ChildAgent(_EchoAgent):
    async def execute(self, *args, **kwargs):
        return await super().execute(*args, **kwargs)


class _FailingPydanticAgent(PydanticAgentBase):
    async def execute_pydantic(self, input_data):
        raise RuntimeError("boom")


class _GroundedTemplate(TemplateAgentV30):
    async def analyze(self, input_data):
        return ("ok", ["10-K 2025"], 0.9, {"next_step": None})


def _run(coro):
    return asyncio.run(coro)


def test_agent_base_reexports_single_schema():
    import core.agents.agent_base as ab
    assert ab.AgentInput is AgentInput and ab.AgentOutput is AgentOutput


def test_error_path_agent_output_is_valid_without_provenance():
    out = AgentOutput(answer="failed", confidence=0.0)
    assert out.provenance_trace == []
    assert out.check_grounding() is False


def test_single_provenance_header_is_wrapped_in_list():
    header = build_provenance_header({"a": 1}, source_data_object="s3://x")
    out = AgentOutput(answer="x", confidence=1.0, provenance_trace=header)
    assert len(out.provenance_trace) == 1 and out.check_grounding()


def test_execute_wrapped_once_per_class_not_per_instance():
    agent = _EchoAgent(config={"agent_id": "echo"})
    assert "execute" not in vars(agent)
    assert getattr(_EchoAgent.execute, "__guarded__", False)


def test_nested_super_execute_runs_guardrails_once():
    agent = _ChildAgent(config={"agent_id": "child"})
    agent.state.logic_layer.active_rules = {"r1": {"==": [1, 1]}}
    _EchoAgent.calls = 0
    _run(agent.execute("q"))
    assert _EchoAgent.calls == 1
    assert len(agent.state.logic_layer.execution_trace) == 1


def test_logic_layer_persists_trace_and_warns_on_failed_rules():
    agent = _EchoAgent(config={"agent_id": "echo"})
    agent.state.logic_layer.active_rules = {"ok": {"==": [1, 1]}, "bad": {"==": [1, 2]}}
    warnings = agent.evaluate_logic_layer()
    trace = agent.state.logic_layer.execution_trace
    assert [t.rule_id for t in trace] == ["ok", "bad"]
    assert len(warnings) == 1 and "bad" in warnings[0]


def test_pydantic_agent_non_standard_mode_returns_full_dump_on_error():
    agent = _FailingPydanticAgent(config={"agent_id": "f"})
    out = _run(agent.execute("query"))
    assert set(out) >= {"answer", "confidence", "metadata", "provenance_trace"}
    assert out["confidence"] == 0.0 and "boom" in out["metadata"]["error"]


def test_pydantic_agent_standard_mode_error_returns_valid_output():
    agent = _FailingPydanticAgent(config={"agent_id": "f"})
    out = _run(agent.execute(AgentInput(query="q")))
    assert isinstance(out, AgentOutput) and out.confidence == 0.0


def test_pending_requests_are_bounded():
    agent = _EchoAgent(config={"agent_id": "echo"})
    agent.MAX_PENDING_REQUESTS = 1
    agent.pending_requests["x"] = None
    assert _run(agent.send_message("peer", {})) is None


@pytest.mark.parametrize("confidence,tier", [(0.9, "autonomous"), (0.6, "hitl_required"), (0.1, "hard_abort")])
def test_template_v30_confidence_tiers(confidence, tier):
    assert TemplateAgentV30(config={"agent_id": "t"}).confidence_tier(confidence) == tier


def test_template_v30_emits_grounded_output():
    out = _run(_GroundedTemplate(config={"agent_id": "t"}).execute(AgentInput(query="q")))
    assert out.check_grounding() and out.metadata["confidence_tier"] == "autonomous"
