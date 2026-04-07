from pydantic import BaseModel, Field
from typing import List, Optional, Union

class CreditMetrics(BaseModel):
    pd: float = Field(..., description="Probabilistic Probability of Default")
    lgd: float = Field(..., description="Deterministic Loss Given Default")
    ead: float = Field(..., description="Exposure at Default")

    @property
    def expected_loss(self) -> float:
        return self.pd * self.lgd * self.ead

class DecisionState(BaseModel):
    conviction_score: float = Field(ge=0, le=1)
    routing_path: str # ["AUTOMATED", "HOTL", "HITL_TIER_3"]
    requires_step_up: bool
    audit_hash: str # SHA-256 of the context + prompt
