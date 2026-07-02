from typing import Any, Dict
from pydantic import BaseModel, Field
from src.pdil.models import ProvenanceHeader


class AgentInput(BaseModel):
    """Deterministic input data and context for agent execution."""
    data: Dict[str, Any] = Field(..., description='Deterministic input data')
    context: str = Field('', description='Additional context for the agent')


class AgentOutput(BaseModel):
    """
    Deterministic output payload and provenance trace for agent execution.
    Provides strict type checking across all agents (Nexus/Sentinel) for horizontal scaling.
    """
    provenance_trace: ProvenanceHeader
    data: Dict[str, Any] = Field(..., description=
        'Deterministic output payload')
    observed_drift: bool = Field(False, description=
        'Flag indicating if logic shifted from existing implementation, triggers self-healing'
        )

    def check_grounding(self) -> bool:
        """
        Verifies that this output contains a valid reference to its source data object,
        satisfying W3C PROV-O compliance requirements.
        """
        return bool(self.provenance_trace.source_data_object)
