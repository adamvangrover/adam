"""
TemplateAgentV30 — the System 2 base class named in AGENTS.md §2.

A thin, documented subclass of ``PydanticAgentBase``. Copy it, override ``analyze``, done:

- I/O: ``AgentInput`` -> ``AgentOutput`` (``core/schemas/agent_schema.py``).
- Provenance: every output carries a PROV-O ``ProvenanceHeader`` in ``provenance_trace``.
- Confidence tiers: thresholds come from ``config/agent_schema.yaml`` and are reported in
  ``metadata["confidence_tier"]`` (``autonomous`` / ``hitl_required`` / ``hard_abort``).
- Harness (``core/engine/harness``): ``h_ctrl`` + ``h_ctx`` run before ``analyze``; ``run_tool`` applies
  the duplicate-call breaker and ``h_act``; ``h_ver`` runs before the output is returned. The turn cap
  counts ``execute`` calls per agent instance (one HNASP trace).
- Routing: never instantiate other agents; request follow-ups via ``metadata["next_step"]``.
"""
import inspect
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Tuple, Union

from core.agents.pydantic_agent_base import PydanticAgentBase
from core.engine.harness import HarnessHalt, HarnessPolicy, h_act, h_ctrl, h_ctx, h_ver, load_harness_policy
from core.engine.harness.policy import POLICY_PATH as _POLICY_PATH
from core.schemas.agent_schema import AgentInput, AgentOutput
from src.pdil.models import build_provenance_header


def load_confidence_tiers(path: Path = _POLICY_PATH) -> Tuple[float, float]:
    """Return (autonomous_min, hitl_min) from the governance policy YAML."""
    policy = load_harness_policy(path)
    return policy.autonomous_min, policy.hitl_min


class TemplateAgentV30(PydanticAgentBase):
    """System 2 template agent. Subclasses override ``analyze``; everything else is inherited."""

    jsonlogic_version = "1.0.0"

    async def analyze(self, input_data: AgentInput) -> Tuple[str, List[str], float, Dict[str, Any]]:
        """Return (answer, sources, confidence, metadata). Override in subclasses."""
        return (f"Template response for: {input_data.query}", [], 0.0, {"next_step": None})

    @property
    def harness_policy(self) -> HarnessPolicy:
        if getattr(self, "_harness_policy", None) is None:
            self._harness_policy = load_harness_policy()
        return self._harness_policy

    @property
    def turn_state(self) -> h_ctrl.TurnState:
        if getattr(self, "_turn_state", None) is None:
            self._turn_state = h_ctrl.TurnState()
        return self._turn_state

    def confidence_tier(self, confidence: float) -> str:
        return h_ver.confidence_tier(confidence, self.harness_policy)

    async def run_tool(self, name: str, arguments: Dict[str, Any],
                       call: Callable[..., Union[Any, Awaitable[Any]]]) -> h_act.ToolResult:
        """Call a deterministic tool under the duplicate-call breaker; normalize its result via ``h_act``."""
        h_ctrl.record_tool_call(self.turn_state, name, arguments, self.harness_policy.control)
        try:
            raw = call(**arguments)
            if inspect.isawaitable(raw):
                raw = await raw
        except Exception as exc:
            raw = exc
        return h_act.post_tool(raw, self.harness_policy.action)

    async def execute_pydantic(self, input_data: AgentInput) -> AgentOutput:
        policy = self.harness_policy
        try:
            h_ctrl.pre_turn(self.turn_state, policy.control)
            input_data, context_report = h_ctx.pre_turn(input_data, policy.context)
            answer, sources, confidence, metadata = await self.analyze(input_data)
        except HarnessHalt as halt:
            return h_ver.hard_abort_output([halt.reason], policy, halt.details)
        header = build_provenance_header(
            {"query": input_data.query, "answer": answer, "sources": sources},
            jsonLogic_version=self.jsonlogic_version,
            confidence_score=confidence,
            derivation_path=f"{type(self).__name__}.analyze",
            source_data_object=sources[0] if sources else "",
        )
        output = AgentOutput(
            answer=answer, sources=sources, confidence=confidence,
            metadata={**metadata, "harness": context_report}, provenance_trace=[header],
        )
        return h_ver.pre_stop(output, policy)
