from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

class LogicLayer(BaseModel):
    """Deterministic Business Logic (JsonLogic AST)."""
    engine: str = "JsonLogic"
    active_rules: Dict[str, Any] = Field(..., description="AST of governance rules")
    state_variables: Dict[str, Any] = Field(..., description="Current variables for rule eval")

class EPAVector(BaseModel):
    """BayesACT Emotional State Vector."""
    evaluation: float = Field(..., ge=-4.3, le=4.3)
    potency: float = Field(..., ge=-4.3, le=4.3)
    activity: float = Field(..., ge=-4.3, le=4.3)

class PersonaState(BaseModel):
    """Probabilistic Personality Definition."""
    model: str = "BayesACT"
    fundamental_identity: EPAVector
    transient_impression: EPAVector
    deflection: float

class AgentState(BaseModel):
    """The HNASP Envelope: The Single Source of Truth."""
    meta_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # The 'Brain': Deterministic Rules
    logic_layer: LogicLayer

    # The 'Heart': Probabilistic Personality
    persona_state: PersonaState

    # The 'Memory': Context Stream
    context_stream: List[Dict[str, Any]]

    class Config:
        json_schema_extra = {
            "description": "A serialized row for the Observation Lakehouse (Delta/Iceberg)."
        }
