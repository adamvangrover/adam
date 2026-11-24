"""Lifecycle harness (core/engine/harness): h_ver, h_ctrl handoffs/turn cap, h_ctx, h_act."""
import ast
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from core.agents.templates import v30_template_agent
from core.agents.templates.v30_template_agent import TemplateAgentV30
from core.engine import harness, meta_orchestrator as mo
from core.engine.harness import HarnessHalt, h_act, h_ctrl, h_ctx, h_ver, load_harness_policy
from core.engine.harness.policy import POLICY_PATH
from core.schemas.agent_schema import AgentInput, AgentOutput
from src.pdil.models import ProvenanceHeader, build_provenance_header

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY = load_harness_policy()


def _header(version="1.0.0"):
    return build_provenance_header({"a": 1}, jsonLogic_version=version, confidence_score=0.9,
                                   derivation_path="test", source_data_object="src")


def _output(confidence=0.95, metadata=None, trace=None):
    return AgentOutput(answer="ok", sources=["src"], confidence=confidence, metadata=metadata or {},
                       provenance_trace=[_header()] if trace is None else trace)


class _Agent(TemplateAgentV30):
    def __init__(self, result=("ok", ["src"], 0.95, {}), version="1.0.0"):
        super().__init__(config={})
        self.result, self.jsonlogic_version = result, version

    async def analyze(self, input_data):
        return self.result


def _run(agent, query="q"):
    return asyncio.run(agent.execute(AgentInput(query=query)))


@pytest.fixture
def orchestrator():
    with patch.object(mo, "LLMPlugin"):
        orch = mo.MetaOrchestrator(legacy_orchestrator=MagicMock())
    orch.swarm_logger, orch.pot_logger = MagicMock(), MagicMock()
    orch._enrich_context = AsyncMock(side_effect=lambda q, c: dict(c or {}))
    orch.config.reflect_on = ()
    return orch


# --- thresholds come from config/agent_schema.yaml --------------------------------------------

def test_policy_reads_thresholds_from_agent_schema_yaml():
    doc = yaml.safe_load(POLICY_PATH.read_text())
    thresholds = doc["governance_policy"]["thresholds"]
    assert POLICY.autonomous_min == thresholds["autonomous_execution"]["min_confidence"]
    assert POLICY.hitl_min == thresholds["human_in_the_loop"]["min_confidence"]
    assert POLICY.control.max_turns == doc["harness"]["control"]["max_turns"]
    assert POLICY.action.max_stdout_chars == doc["harness"]["action"]["max_stdout_chars"]


def test_policy_overrides_follow_the_yaml(tmp_path):
    doc = yaml.safe_load(POLICY_PATH.read_text())
    doc["harness"]["control"]["max_turns"] = 2
    path = tmp_path / "agent_schema.yaml"
    path.write_text(yaml.safe_dump(doc))
    assert load_harness_policy(path).control.max_turns == 2


# --- (a) missing jsonLogic_version / provenance -> hard abort below the HITL floor ------------

@pytest.mark.parametrize("trace", [[], [ProvenanceHeader.model_construct(confidence_score=0.9)],
                                   [_header(version="")]])
def test_h_ver_blocks_exit_without_provenance_or_jsonlogic_version(trace):
    out = h_ver.pre_stop(_output(confidence=0.99, metadata={"next_step": "invoke_surveillance"}, trace=trace),
                         POLICY)
    assert out.confidence < POLICY.hitl_min
    assert out.metadata["confidence_tier"] == "hard_abort"
    assert out.metadata["event"] == h_ver.CONVICTION_BREACH_EVENT
    assert "next_step" not in out.metadata
    assert out.provenance_trace == []


def test_template_agent_without_jsonlogic_version_aborts():
    out = _run(_Agent(version=""))
    assert out.confidence < POLICY.hitl_min
    assert out.metadata["harness_violations"] == ["missing_jsonLogic_version"]


def test_grounded_output_passes_verification():
    out = _run(_Agent())
    assert out.metadata["confidence_tier"] == "autonomous"
    assert out.metadata["hitl_required"] is False
    assert out.provenance_trace[0].jsonLogic_version == "1.0.0"


# --- (b) metadata.next_step handoffs; no direct peer instantiation ----------------------------

def test_next_step_handoff_is_dispatched_by_orchestrator(orchestrator):
    surveillance = AsyncMock(return_value=_output(metadata={"source": "surveillance"}))
    orchestrator._route_registry[mo.Route.ESG] = AsyncMock(
        return_value=_output(metadata={"next_step": "invoke_surveillance", "query": "watch ACME"}))
    orchestrator._route_registry[mo.Route.SURVEILLANCE] = surveillance

    result = asyncio.run(orchestrator.route_request("esg review", {"force_route": "esg"}))

    assert result.metadata["source"] == "surveillance"
    query, context = surveillance.await_args.args
    assert query == "watch ACME"
    assert context["handoff_from"] == "invoke_surveillance"


def test_unroutable_next_step_is_left_for_the_caller(orchestrator):
    orchestrator._route_registry[mo.Route.ESG] = AsyncMock(return_value=_output(metadata={"next_step": "update_hdkg"}))
    result = asyncio.run(orchestrator.route_request("esg review", {"force_route": "esg"}))
    assert result.metadata["next_step"] == "update_hdkg"


def test_hitl_output_is_not_auto_handed_off(orchestrator):
    surveillance = AsyncMock()
    orchestrator._route_registry[mo.Route.ESG] = AsyncMock(return_value=_output(
        confidence=0.6, metadata={"next_step": "invoke_surveillance"}))
    orchestrator._route_registry[mo.Route.SURVEILLANCE] = surveillance
    result = asyncio.run(orchestrator.route_request("esg review", {"force_route": "esg"}))
    assert result.metadata["hitl_required"] is True
    surveillance.assert_not_awaited()


def _instantiated_agent_classes(path: Path):
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name.endswith("Agent") or name.endswith("AgentV30"):
                names.add(name)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("core.agents."):
            if node.module not in {"core.agents.pydantic_agent_base", "core.agents.agent_base"}:
                names.add(node.module)
    return names


@pytest.mark.parametrize("path", [Path(v30_template_agent.__file__),
                                  *sorted(Path(harness.__file__).parent.glob("*.py"))],
                         ids=lambda p: p.name)
def test_template_and_harness_never_instantiate_peer_agents(path):
    assert _instantiated_agent_classes(path) == set()


# --- (c) high-impact actions require HITL even at >= 0.85 --------------------------------------

@pytest.mark.parametrize("metadata", [{"action": POLICY.verification.high_impact_actions[0]},
                                      {"impact": POLICY.verification.hitl_impact_levels[0]},
                                      {"next_step": POLICY.verification.high_impact_actions[-1]}])
def test_high_impact_action_forces_hitl_at_high_confidence(metadata):
    out = _run(_Agent(result=("commit", ["src"], 0.99, metadata)))
    assert out.confidence >= POLICY.autonomous_min
    assert out.metadata["hitl_required"] is True
    assert out.metadata["high_impact"] is True
    assert out.metadata["confidence_tier"] == "hitl_required"


def test_low_impact_action_stays_autonomous():
    out = h_ver.pre_stop(_output(metadata={"action": "summarize"}), POLICY)
    assert out.metadata["hitl_required"] is False


# --- (d) h_ctrl turn cap and duplicate-call breaker terminate runaway loops ---------------------

def test_turn_cap_halts_after_max_turns():
    state = h_ctrl.TurnState()
    for _ in range(POLICY.control.max_turns):
        h_ctrl.pre_turn(state, POLICY.control)
    with pytest.raises(HarnessHalt) as exc:
        h_ctrl.pre_turn(state, POLICY.control)
    assert exc.value.reason == "turn_cap_exceeded"


def test_template_agent_turn_cap_aborts_runaway_execution():
    agent = _Agent()
    outs = [_run(agent, query=f"q{i}") for i in range(POLICY.control.max_turns + 1)]
    assert all(o.metadata["confidence_tier"] == "autonomous" for o in outs[:-1])
    assert outs[-1].metadata["harness_violations"] == ["turn_cap_exceeded"]
    assert outs[-1].confidence < POLICY.hitl_min


def test_orchestrator_terminates_self_handoff_loop(orchestrator):
    calls = {"n": 0}

    async def runaway(query, context):
        calls["n"] += 1
        return _output(metadata={"next_step": "invoke_esg", "query": f"again {calls['n']}"})

    orchestrator._route_registry[mo.Route.ESG] = runaway
    result = asyncio.run(orchestrator.route_request("esg review", {"force_route": "esg"}))
    assert result["status"] == "harness_halt"
    assert result["reason"] == "turn_cap_exceeded"
    assert calls["n"] == POLICY.control.max_turns


def test_duplicate_tool_call_breaker():
    agent = _Agent()
    tool = MagicMock(return_value="42")
    for _ in range(POLICY.control.max_duplicate_tool_calls):
        assert asyncio.run(agent.run_tool("calc", {"x": 1}, tool)).ok
    with pytest.raises(HarnessHalt) as exc:
        asyncio.run(agent.run_tool("calc", {"x": 1}, tool))
    assert exc.value.reason == "duplicate_tool_call"


# --- h_ctx / h_act --------------------------------------------------------------------------------

def test_h_ctx_masks_stale_observations_and_enforces_budget():
    cfg = POLICY.context
    obs = [f"obs{i}" for i in range(cfg.keep_last_observations + 2)]
    masked, _ = h_ctx.pre_turn(AgentInput(query="q", context={cfg.observations_key: obs}), cfg)
    kept = masked.context[cfg.observations_key]
    assert kept[:2] == [cfg.masked_placeholder] * 2
    assert kept[2:] == obs[2:]
    huge = "x" * (cfg.max_context_tokens * cfg.chars_per_token + cfg.chars_per_token)
    with pytest.raises(HarnessHalt) as exc:
        h_ctx.pre_turn(AgentInput(query="q", context={"blob": huge}), cfg)
    assert exc.value.reason == "context_budget_exceeded"


def test_h_act_truncates_stdout_and_normalizes_errors():
    cfg = POLICY.action
    res = h_act.post_tool("y" * (cfg.max_stdout_chars * 2), cfg)
    assert res.truncated and len(res.stdout) < cfg.max_stdout_chars * 2
    err = h_act.post_tool(ValueError("bad input"), cfg)
    assert not err.ok and err.error["type"] == "ValueError"


def test_h_ctx_counts_non_ascii_characters_once():
    cfg = POLICY.context
    assert h_ctx.estimate_tokens("中" * cfg.chars_per_token * 10, cfg.chars_per_token) <= 11
