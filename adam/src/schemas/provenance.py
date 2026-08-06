from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import hashlib
import json

class ProvenanceHeader(BaseModel):
    """
    Immutable structured narrative logging header for agent outputs.
    Ensures that every agent decision can be traced back to its specific execution context.
    """
    git_commit_hash: str = Field(
        ...,
        description="The specific Git commit hash of the codebase at execution time."
    )
    jsonLogic_version: str = Field(
        ...,
        description="The version of the jsonLogic ruleset applied for this decision."
    )
    source_data_reference: str = Field(
        ...,
        description="Reference/URI to the exact source data object used for grounding."
    )
    hash: Optional[str] = Field(
        None,
        description="Immutable cryptographic hash of the execution context."
    )

    def model_post_init(self, __context: Any) -> None:
        """Automatically generate an immutable hash of the header if not provided."""
        if not self.hash:
            data = f"{self.git_commit_hash}:{self.jsonLogic_version}:{self.source_data_reference}"
            # Because frozen=True, we have to bypass normal assignment
            object.__setattr__(self, 'hash', hashlib.sha256(data.encode('utf-8')).hexdigest())

    model_config = {
        "frozen": True  # Enforce immutability to satisfy audit trail requirements
    }

class AgentOutput(BaseModel):
    """
    Standard envelope for all outputs from System 1 (Agents) to System 2 (Deterministic).
    Requires a valid ProvenanceHeader for W3C PROV-O compliance.
    """
    provenance: ProvenanceHeader = Field(
        ...,
        description="Immutable W3C PROV-O compliant provenance metadata."
    )
    answer: str = Field(..., description="The final synthesized answer or decision.")
    sources: List[str] = Field(
        default_factory=list,
        description="List of citations (filenames, URLs) used in reasoning."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Conviction score (0.0 to 1.0). Must be >= 0.5 per Sentinel rules."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Debug info, token usage, next_step routing."
    )

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if v < 0.5:
            raise ValueError("Confidence score must be >= 0.5 per Sentinel security rules.")
        return v
