from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field, ConfigDict, UUID4
from datetime import datetime

# --- Meta Namespace ---

class ModelConfig(BaseModel):
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None

class SecurityContext(BaseModel):
    clearance: str = "public"
    user_id: str
    roles: List[str] = Field(default_factory=list)

class Meta(BaseModel):
    agent_id: str
    trace_id: str
    timestamp: datetime
    model_config_data: ModelConfig = Field(alias="model_config")
    security_context: SecurityContext

    model_config = ConfigDict(populate_by_name=True)

# --- Persona State Namespace (BayesACT) ---

class EPAVector(BaseModel):
    E: float = Field(..., description="Evaluation: Good (positive) vs Bad (negative)")
    P: float = Field(..., description="Potency: Powerful (positive) vs Powerless (negative)")
    A: float = Field(..., description="Activity: Active (positive) vs Passive (negative)")

class Identity(BaseModel):
    label: str
    fundamental_epa: Optional[EPAVector] = None
    transient_epa: Optional[EPAVector] = None
    uncertainty_covariance: Optional[List[float]] = None # Simplified representation
    confidence: Optional[float] = None

class PersonaIdentities(BaseModel):
    self: Identity
    user: Identity

class PersonaDynamics(BaseModel):
    current_deflection: float
    target_behavior_epa: Optional[EPAVector] = None

class PersonaState(BaseModel):
    model: Literal["BayesACT"] = "BayesACT"
    identities: PersonaIdentities
    dynamics: PersonaDynamics

# --- Logic Layer Namespace (JsonLogic) ---

class Rule(BaseModel):
    # JsonLogic rules are recursive dictionaries
    # We use Dict[str, Any] because defining the recursive JsonLogic schema in Pydantic is complex
    rule_def: Dict[str, Any]

class ActiveRules(BaseModel):
    # Map of rule_id to rule definition (which is a dict representing JsonLogic)
    # The whitepaper shows "active_rules": { "rule_name": { "if": ... } }
    rules: Dict[str, Dict[str, Any]]

    # We use a custom root or just a dict, but Pydantic models are cleaner.
    # However, since the keys are dynamic rule names, we might just use a Dict or a root model.
    # Let's use Dict[str, Dict[str, Any]] but mapped in the LogicLayer model.

class ExecutionTrace(BaseModel):
    rule_id: str
    result: Any
    step_by_step: Optional[List[Dict[str, Any]]] = None

class LogicLayer(BaseModel):
    engine: Literal["JsonLogic"] = "JsonLogic"
    version: str = "2.0"
    state_variables: Dict[str, Any]
    active_rules: Dict[str, Dict[str, Any]]
    execution_trace: Optional[ExecutionTrace] = None

# --- Context Stream Namespace ---

class Turn(BaseModel):
    role: Literal["user", "assistant", "agent_thought", "system"]
    timestamp: datetime
    content: Optional[str] = None
    intent: Optional[str] = None
    sentiment_vector: Optional[List[float]] = None
    logic_eval: Optional[Dict[str, Any]] = None # For agent_thought
    internal_monologue: Optional[str] = None # For agent_thought

class ContextStream(BaseModel):
    window_id: int
    turns: List[Turn]

# --- Root HNASP Schema ---

class HNASP(BaseModel):
    meta: Meta
    persona_state: PersonaState
    logic_layer: LogicLayer
    context_stream: ContextStream

    model_config = ConfigDict(populate_by_name=True)
