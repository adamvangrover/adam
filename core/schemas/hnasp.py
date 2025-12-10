from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Literal, Optional
from datetime import datetime
import uuid

class Meta(BaseModel):
    agent_id: str
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    security_context: Dict[str, Any] = Field(default_factory=dict)

class PersonaDynamics(BaseModel):
    current_deflection: float = 0.0
    optimal_behavior_epa: Optional[List[float]] = None

class PersonaState(BaseModel):
    identities: Dict[str, Dict[str, List[float]]] = Field(
        default_factory=lambda: {
            "self": {"fundamental_epa": [0.0, 0.0, 0.0], "transient_epa": [0.0, 0.0, 0.0]},
            "user": {"fundamental_epa": [0.0, 0.0, 0.0], "transient_epa": [0.0, 0.0, 0.0]}
        }
    )
    dynamics: PersonaDynamics = Field(default_factory=PersonaDynamics)

class LogicLayer(BaseModel):
    engine: Literal["JsonLogic"] = "JsonLogic"
    state_variables: Dict[str, Any] = Field(default_factory=dict)
    active_rules: Dict[str, Any] = Field(default_factory=dict)
    execution_trace: List[Any] = Field(default_factory=list)

class Turn(BaseModel):
    role: Literal["user", "agent", "thought"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ContextStream(BaseModel):
    turns: List[Turn] = Field(default_factory=list)

class HNASPState(BaseModel):
    meta: Meta
    persona: PersonaState
    logic_layer: LogicLayer
    context_stream: ContextStream
