# core/schemas/hnasp.py

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

# --- Enums / Constants ---

# --- Sub-Schemas ---

class SecurityContext(BaseModel):
    user_id: str
    clearance: str

class Meta(BaseModel):
    agent_id: str
    trace_id: str
    security_context: SecurityContext

class EPAVector(BaseModel):
    E: float
    P: float
    A: float

class Identity(BaseModel):
    label: str
    fundamental_epa: EPAVector
    transient_epa: Optional[EPAVector] = None

class PersonaIdentities(BaseModel):
    self: Identity
    user: Identity

class PersonaDynamics(BaseModel):
    current_deflection: float = 0.0

class PersonaState(BaseModel):
    identities: PersonaIdentities
    dynamics: PersonaDynamics = Field(default_factory=PersonaDynamics)

class LogicLayer(BaseModel):
    engine: str = "JsonLogic"
    active_rules: Dict[str, Any] = Field(default_factory=dict)
    state_variables: Dict[str, Any] = Field(default_factory=dict)
    execution_trace: List[Any] = Field(default_factory=list)

class ExecutionTrace(BaseModel):
    rule_id: str
    result: Any
    error: Optional[str] = None
    timestamp: datetime

class Turn(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    logic_eval: Optional[Dict[str, Any]] = None

class ContextStream(BaseModel):
    turns: List[Turn] = Field(default_factory=list)

class ModelConfig(BaseModel):
    """Configuration for the HNASP model."""
    provider: str = "default"
    temperature: float = 0.7
    max_tokens: int = 1000

# --- Main HNASP State ---

class HNASPState(BaseModel):
    """
    Hybrid Neurosymbolic Agent State Protocol (HNASP) Schema.
    Treats agent state as a structured database row.
    """
    meta: Meta
    logic_layer: LogicLayer
    persona_state: PersonaState
    context_stream: ContextStream
    model_config_data: Optional[ModelConfig] = None

    model_config = ConfigDict(populate_by_name=True)
