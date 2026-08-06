import json
import hashlib
from typing import Any, Dict
from pydantic import BaseModel, Field
import hashlib
import json
from src.pdil.models import ProvenanceHeader


def compute_deterministic_hash(data: dict) -> str:
    """
    Standardizes dictionary hashing by serializing with sort_keys=True and
    separators=(',', ':'), then returning its SHA-256 hexdigest.
    Used for ensuring provenance trace and event sourcing integrity.
    """
    serialized = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


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

def compute_deterministic_hash(data: dict) -> str:
    """
    Serializes a dictionary into a JSON string with sorted keys and returns its SHA-256 hash.
    Used for ensuring provenance trace integrity.
    """
    payload_json = json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload_json).hexdigest()
