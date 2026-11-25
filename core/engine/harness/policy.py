"""Harness policy: every hook threshold is loaded from ``config/agent_schema.yaml``."""
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

POLICY_PATH = Path(__file__).resolve().parents[3] / "config" / "agent_schema.yaml"


class HarnessHalt(Exception):
    """Raised by a hook to stop the current loop; callers turn it into a hard-abort result."""

    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}

    def as_result(self) -> Dict[str, Any]:
        return {"status": "harness_halt", "reason": self.reason, "details": self.details,
                "confidence_tier": "hard_abort", "hitl_required": False}


class ContextPolicy(BaseModel):
    observations_key: str
    keep_last_observations: int = Field(..., ge=0)
    masked_placeholder: str
    max_context_tokens: int = Field(..., gt=0)
    chars_per_token: int = Field(..., gt=0)


class ControlPolicy(BaseModel):
    max_turns: int = Field(..., gt=0)
    max_duplicate_tool_calls: int = Field(..., gt=0)


class ActionPolicy(BaseModel):
    max_stdout_chars: int = Field(..., gt=0)
    truncation_marker: str


class VerificationPolicy(BaseModel):
    abort_confidence: float = Field(..., ge=0.0, le=1.0)
    hitl_impact_levels: List[str]
    high_impact_actions: List[str]


class HarnessPolicy(BaseModel):
    autonomous_min: float = Field(..., ge=0.0, le=1.0)
    hitl_min: float = Field(..., ge=0.0, le=1.0)
    context: ContextPolicy
    control: ControlPolicy
    action: ActionPolicy
    verification: VerificationPolicy

    @model_validator(mode="after")
    def _ordered_tiers(self) -> "HarnessPolicy":
        if not self.verification.abort_confidence < self.hitl_min <= self.autonomous_min:
            raise ValueError("require abort_confidence < hitl_min <= autonomous_min")
        return self


def load_harness_policy(path: Path = POLICY_PATH) -> HarnessPolicy:
    """Parse confidence tiers and the ``harness`` section of the governance policy YAML."""
    doc = yaml.safe_load(Path(path).read_text())
    thresholds = doc["governance_policy"]["thresholds"]
    return HarnessPolicy(
        autonomous_min=thresholds["autonomous_execution"]["min_confidence"],
        hitl_min=thresholds["human_in_the_loop"]["min_confidence"],
        **doc["harness"],
    )
