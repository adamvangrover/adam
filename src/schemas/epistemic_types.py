from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field

class EpistemicStatus(str, Enum):
    KNOWN = "KNOWN"
    SUPPORTED = "SUPPORTED"
    UNCERTAIN = "UNCERTAIN"
    CONFLICTED = "CONFLICTED"
    OOD = "OUT_OF_DISTRIBUTION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNRESOLVED = "UNRESOLVED"
    UNKNOWN = "UNKNOWN"

class TemporalBounds(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    event_time_te: datetime = Field(..., description="t_e: Event Time")
    knowledge_time_tk: datetime = Field(..., description="t_k: Knowledge Time")
    decision_time_td: Optional[datetime] = Field(None, description="t_d: Decision Time")
    execution_time_tx: Optional[datetime] = Field(None, description="t_x: Execution Time")
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

class AuthorityBoundaryRecord(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    record_type: str = Field(default="AuthorityBoundaryRecord")
    schema_version: str = Field(default="2.0.0")
    authority: str = Field(..., description="Must be 'NONE'")
    subject: Dict[str, Any]
    source_lineage: Dict[str, Any]
    epistemic_input: Dict[str, Any]
    model_context: Dict[str, Any]
    temporal_bounds: TemporalBounds
    provenance: Dict[str, Any]
    admission_criteria: Dict[str, bool]
