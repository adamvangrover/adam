from typing import Any, Dict
from pydantic import BaseModel, Field
from src.governance.gatekeeper import ProvenanceHeader

class AgentInput(BaseModel):
    data: Dict[str, Any] = Field(..., description="Deterministic input data")
    context: str = Field("", description="Additional context for the agent")

class AgentOutput(BaseModel):
    provenance_trace: ProvenanceHeader = Field(..., description="W3C PROV-O compliant provenance trace")
    data: Dict[str, Any] = Field(..., description="Deterministic output payload")
    observed_drift: bool = Field(False, description="Flag indicating if logic shifted from existing implementation")
