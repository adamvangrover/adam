import datetime
import uuid
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class EventEnvelope(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex}")
    event_type: str
    occurred_at: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    actor_id: str
    correlation_id: str
    causation_id: Optional[str] = None
    context_id: Optional[str] = None
    context_hash: Optional[str] = None
    policy_version: Optional[str] = None
    code_revision: Optional[str] = None
    risk_class: str
    payload_hash: Optional[str] = None
    provenance_id: Optional[str] = None
    schema_version: int = 1
    data: Dict[str, Any]
