from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
import uuid

from pydantic import BaseModel, Field, ConfigDict

# --- Meta Namespace ---

class ModelConfig(BaseModel):
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None

class SecurityContext(BaseModel):
    """
    Explicit security context for RBAC.
    """
    clearance: str = "public"
    user_id: str
    roles: List[str] = Field(default_factory=list)

class Meta(BaseModel):
    agent_id: str
    # MERGE NOTE: Restored default factories from 'refactor' for ease of instantiation
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_config_data: ModelConfig = Field(default_factory=lambda: ModelConfig(model="gpt-4"), alias="model_config")
    security_context: SecurityContext

    model_config = ConfigDict(populate_by_name=True)

# --- Persona State Namespace (BayesACT) ---

class EPAVector(BaseModel):
    """
    Fundamental unit for Affect Control Theory.
    """
    E: float = Field(..., description="Evaluation: Good (positive) vs Bad (negative)")
    P: float = Field(..., description="Potency: Powerful (positive) vs Powerless (negative)")
    A: float = Field(..., description="Activity: Active (positive) vs Passive (negative)")

class Identity(BaseModel):
    label: str
    fundamental_epa: Optional[EPAVector] = None
    transient_epa: Optional[EPAVector] = None
    uncertainty_covariance: Optional[List[float]] = None 
    confidence: Optional[float] = None

class PersonaIdentities(BaseModel):
    self: Identity
    user: Identity

class PersonaDynamics(BaseModel):
    current_deflection: float = 0.0
    target_behavior_epa: Optional[EPAVector] = None

class PersonaState(BaseModel):
    model_type: Literal["BayesACT"] = "BayesACT"
    identities: PersonaIdentities
    dynamics: PersonaDynamics = Field(default_factory=lambda: PersonaDynamics(current_deflection=0.0))

# --- Logic Layer Namespace (JsonLogic) ---

class ExecutionTrace(BaseModel):
    rule_id: str
    result: Any
    step_by_step: Optional[List[Dict[str, Any]]] = None

class LogicLayer(BaseModel):
    engine: Literal["JsonLogic"] = "JsonLogic"
    version: str = "2.0"
    state_variables: Dict[str, Any] = Field(default_factory=dict)
    active_rules: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    execution_trace: Optional[ExecutionTrace] = None

# --- Context Stream Namespace ---

class Turn(BaseModel):
    # MERGE NOTE: Adopted the richer role set from 'main'
    role: Literal["user", "assistant", "agent_thought", "system"]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    content: Optional[str] = None
    intent: Optional[str] = None
    sentiment_vector: Optional[List[float]] = None
    logic_eval: Optional[Dict[str, Any]] = None # Specific to 'agent_thought'
    internal_monologue: Optional[str] = None    # Specific to 'agent_thought'

class ContextStream(BaseModel):
    window_id: int = 0
    turns: List[Turn] = Field(default_factory=list)

# --- Root HNASP Schema ---

class HNASPState(BaseModel):
    meta: Meta
    persona_state: PersonaState
    logic_layer: LogicLayer
    context_stream: ContextStream

    model_config = ConfigDict(populate_by_name=True)