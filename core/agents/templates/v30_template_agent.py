"""
TemplateAgentV30 — the System 2 base class named in AGENTS.md §2.

A thin, documented subclass of ``PydanticAgentBase``. Copy it, override ``analyze``, done:

- I/O: ``AgentInput`` -> ``AgentOutput`` (``core/schemas/agent_schema.py``).
- Provenance: every output carries a PROV-O ``ProvenanceHeader`` in ``provenance_trace``.
- Confidence tiers: thresholds come from ``config/agent_schema.yaml`` and are reported in
  ``metadata["confidence_tier"]`` (``autonomous`` / ``hitl_required`` / ``hard_abort``).
- Routing: never instantiate other agents; request follow-ups via ``metadata["next_step"]``.
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from core.agents.pydantic_agent_base import PydanticAgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput
from src.pdil.models import build_provenance_header

_POLICY_PATH = Path(__file__).resolve().parents[3] / "config" / "agent_schema.yaml"


def load_confidence_tiers(path: Path = _POLICY_PATH) -> Tuple[float, float]:
    """Return (autonomous_min, hitl_min) from the governance policy YAML."""
    thresholds = yaml.safe_load(path.read_text())["governance_policy"]["thresholds"]
    return (
        float(thresholds["autonomous_execution"]["min_confidence"]),
        float(thresholds["human_in_the_loop"]["min_confidence"]),
    )


class TemplateAgentV30(PydanticAgentBase):
    """System 2 template agent. Subclasses override ``analyze``; everything else is inherited."""

    jsonlogic_version = "1.0.0"

    async def analyze(self, input_data: AgentInput) -> Tuple[str, List[str], float, Dict[str, Any]]:
        """Return (answer, sources, confidence, metadata). Override in subclasses."""
        return (f"Template response for: {input_data.query}", [], 0.0, {"next_step": None})

    def confidence_tier(self, confidence: float) -> str:
        autonomous_min, hitl_min = load_confidence_tiers()
        if confidence >= autonomous_min:
            return "autonomous"
        if confidence >= hitl_min:
            return "hitl_required"
        return "hard_abort"

    async def execute_pydantic(self, input_data: AgentInput) -> AgentOutput:
        answer, sources, confidence, metadata = await self.analyze(input_data)
        header = build_provenance_header(
            {"query": input_data.query, "answer": answer, "sources": sources},
            jsonLogic_version=self.jsonlogic_version,
            confidence_score=confidence,
            derivation_path=f"{type(self).__name__}.analyze",
            source_data_object=sources[0] if sources else "",
        )
        metadata = {**metadata, "confidence_tier": self.confidence_tier(confidence)}
        return AgentOutput(
            answer=answer, sources=sources, confidence=confidence,
            metadata=metadata, provenance_trace=[header],
        )
