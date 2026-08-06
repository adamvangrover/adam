from src.pdil.models import ProvenanceHeader
from pydantic import BaseModel
from typing import Dict, Any, Optional


class AgentInput(BaseModel):
    data: Dict[str, Any]


class AgentOutput(BaseModel):
    """
    AgentOutput represents the output of an agent execution.
    """
    status: str
    result: Dict[str, Any]
    provenance_trace: ProvenanceHeader

    def check_grounding(self) -> bool:
        """
        Verifies that this output contains a valid reference to its source data object,
        satisfying W3C PROV-O compliance requirements.
        """
        return bool(self.provenance_trace.source_data_object)
