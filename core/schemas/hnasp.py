from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
import uuid

class EPAVector(BaseModel):
    E: float = 0.0
    P: float = 0.0
    A: float = 0.0

class Identity(BaseModel):
    label: str
    fundamental_epa: Optional[EPAVector] = None
    transient_epa: Optional[EPAVector] = None

class PersonaIdentities(BaseModel):
    self: Identity
    user: Identity

class PersonaDynamics(BaseModel):
    current_deflection: float = 0.0
    target_behavior_epa: Optional[EPAVector] = None # Added for BayesACTEngine compatibility

    model_config = ConfigDict(extra='allow') # Allow dynamic attributes

class PersonaState(BaseModel):
    identities: PersonaIdentities
    dynamics: PersonaDynamics = Field(default_factory=PersonaDynamics)

class SecurityContext(BaseModel):
    user_id: str
    clearance: str

class Meta(BaseModel):
    agent_id: str
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    security_context: SecurityContext
    timestamp: Optional[datetime] = None # Added for run_cycle compatibility

    model_config = ConfigDict(extra='allow') # Allow dynamic attributes

class ExecutionTrace(BaseModel):
    rule_id: str
    result: Any
    timestamp: datetime
    error: Optional[str] = None

class LogicLayer(BaseModel):
    state_variables: Dict[str, Any] = Field(default_factory=dict)
    active_rules: Dict[str, Any] = Field(default_factory=dict)
    execution_trace: List[ExecutionTrace] = Field(default_factory=list)

class Turn(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    intent: Optional[str] = None
    sentiment_vector: Optional[EPAVector] = None
    internal_monologue: Optional[str] = None

class ModelConfig(BaseModel):
    model_name: str = "default"
    temperature: float = 0.7

    model_config = ConfigDict(populate_by_name=True)

class ContextStream(BaseModel):
    # Field alias required because code expects "turns" but tests or legacy code might populate "history"
    history: List[Turn] = Field(default_factory=list, alias="turns")

    model_config = ConfigDict(populate_by_name=True, extra='allow')

    @property
    def turns(self):
        return self.history

    @turns.setter
    def turns(self, value):
        self.history = value

class HNASPState(BaseModel):
    meta: Meta
    persona_state: PersonaState
    logic_layer: LogicLayer
    context_stream: ContextStream

# Deprecated or Alias for Backward Compatibility
HNASP = HNASPState
