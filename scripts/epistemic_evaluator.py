from pydantic import BaseModel, ConfigDict
from enum import Enum
from typing import Optional

class EpistemicState(str, Enum):
    SUPPORTED = "SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTED = "CONFLICTED"
    UNKNOWN = "UNKNOWN"

class EntityEvaluation(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')

    entity_id: str
    t_e: int
    t_k: int
    t_d: int
    t_x: int
    authority: str
    divergence: float
    tau_w: float
    provo_traces_intact: bool

class EvaluationResult(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')

    passed: bool
    state: EpistemicState
    reason: Optional[str] = None

def evaluate_entity(entity: EntityEvaluation) -> EvaluationResult:
    if entity.authority != "NONE":
         return EvaluationResult(passed=False, state=EpistemicState.UNKNOWN, reason="Authority Violation: Authority must be NONE.")

    if not (entity.t_e <= entity.t_k <= entity.t_d <= entity.t_x):
        return EvaluationResult(passed=False, state=EpistemicState.UNKNOWN, reason="Temporal Check: Expected t_e <= t_k <= t_d <= t_x.")

    if not entity.provo_traces_intact:
         return EvaluationResult(passed=False, state=EpistemicState.UNKNOWN, reason="Provenance: W3C PROV-O traces must be intact.")

    if entity.divergence > entity.tau_w:
         return EvaluationResult(passed=False, state=EpistemicState.CONFLICTED, reason="Divergence exceeds tau_w.")

    return EvaluationResult(passed=True, state=EpistemicState.SUPPORTED)
