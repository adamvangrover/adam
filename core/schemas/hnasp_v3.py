from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal, Union
from datetime import datetime
from enum import Enum
import uuid

from pydantic import BaseModel, Field, ConfigDict, computed_field

# ==========================================
# 1. CORE PRIMITIVES (Shared Across Namespaces)
# ==========================================

class EPAVector(BaseModel):
    """
    The fundamental unit of Affect Control Theory (BayesACT).
    Represents a point in 3D socio-emotional space.
    Range usually [-4.3, +4.3].
    """
    E: float = Field(..., description="Evaluation: Good (positive) vs Bad (negative)")
    P: float = Field(..., description="Potency: Powerful (positive) vs Powerless (negative)")
    A: float = Field(..., description="Activity: Active (positive) vs Passive (negative)")

    def to_list(self) -> List[float]:
        return [self.E, self.P, self.A]

class RoleEnum(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    AGENT_THOUGHT = "agent_thought" # For Chain of Thought logging
    TOOL_OUTPUT = "tool_output"

# ==========================================
# 2. META NAMESPACE (System & Security)
# ==========================================

class SecurityContext(BaseModel):
    """
    Explicit security context for RBAC and Data Governance.
    """
    user_id: str
    clearance_level: Literal["public", "internal", "confidential", "top_secret"] = "public"
    roles: List[str] = Field(default_factory=list)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class ModelConfig(BaseModel):
    """
    Configuration for the inference engine.
    """
    provider: str = "openai" # e.g., openai, anthropic, local
    model_name: str = "gpt-4-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0

class Meta(BaseModel):
    """
    The 'Header' of the state. Tracks provenance and config.
    """
    agent_id: str
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "3.0.0"
    
    # Operational Telemetry
    latency_ms: Optional[float] = None
    token_usage: Dict[str, int] = Field(default_factory=lambda: {"prompt": 0, "completion": 0})
    
    model_config_data: ModelConfig = Field(default_factory=ModelConfig, alias="model_config")
    security_context: SecurityContext

    model_config = ConfigDict(populate_by_name=True)

# ==========================================
# 3. PERSONA STATE NAMESPACE (BayesACT Engine)
# ==========================================

class Identity(BaseModel):
    """
    Represents a social identity within the Affect Control Theory framework.
    """
    label: str = Field(..., description="The semantic label, e.g., 'Advisor', 'Client'")
    
    # The 'True' identity (Who I usually am)
    fundamental_epa: EPAVector 
    
    # The 'Current' impression (Who I seem to be right now due to recent events)
    transient_epa: Optional[EPAVector] = None
    
    # Covariance matrix for uncertainty (BayesACT specific)
    uncertainty_covariance: Optional[List[List[float]]] = None 
    confidence: float = 1.0

class PersonaDynamics(BaseModel):
    """
    Tracks the emotional tension and trajectory of the interaction.
    """
    current_deflection: float = Field(0.0, description="The 'Stress' or error between Fundamental and Transient states.")
    optimal_behavior_epa: Optional[EPAVector] = Field(None, description="The calculated next best action to minimize deflection.")
    emotional_state_label: Optional[str] = Field(None, description="Human readable emotion derived from EPA (e.g., 'Anxious', 'Authoritative')")

class PersonaState(BaseModel):
    model_type: Literal["BayesACT"] = "BayesACT"
    
    # Who is in the room?
    self_identity: Identity
    user_identity: Identity
    
    dynamics: PersonaDynamics = Field(default_factory=PersonaDynamics)

# ==========================================
# 4. LOGIC LAYER NAMESPACE (Symbolic/Deterministic)
# ==========================================

class ExecutionTrace(BaseModel):
    rule_id: str
    result: Any
    step_by_step: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None

class MemoryStore(BaseModel):
    """
    Differentiates types of symbolic memory.
    """
    working_memory: Dict[str, Any] = Field(default_factory=dict, description="Transient variables for the current task")
    long_term_facts: Dict[str, Any] = Field(default_factory=dict, description="Persisted user facts")

class LogicLayer(BaseModel):
    engine: Literal["JsonLogic", "PythonExpr"] = "JsonLogic"
    version: str = "2.1"
    
    memory: MemoryStore = Field(default_factory=MemoryStore)
    
    # Flattened access for backwards compatibility if needed
    @computed_field
    def state_variables(self) -> Dict[str, Any]:
        return {**self.memory.long_term_facts, **self.memory.working_memory}

    active_rules: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="The JsonLogic rules currently in scope")
    execution_trace: List[ExecutionTrace] = Field(default_factory=list)

# ==========================================
# 5. CONTEXT STREAM NAMESPACE (Narrative History)
# ==========================================

class Turn(BaseModel):
    """
    A single atom of conversation.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: RoleEnum
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # The actual text
    content: Optional[str] = None
    
    # CoT / Reasoning (Hidden from user, visible to system)
    internal_monologue: Optional[str] = None
    
    # Semantic Metadata
    intent: Optional[str] = None
    sentiment: Optional[EPAVector] = Field(None, description="The EPA rating of this specific text utterance")
    
    # Tool usage specifics
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ContextStream(BaseModel):
    window_id: int = 0
    max_turns: int = 50
    turns: List[Turn] = Field(default_factory=list)

    def add_turn(self, turn: Turn):
        self.turns.append(turn)
        # Simple sliding window logic could go here

# ==========================================
# 6. ROOT HNASP SCHEMA
# ==========================================

class HNASPState(BaseModel):
    """
    Hierarchical Narrative Agent State Protocol (v3.0)
    
    A standardized envelope for agent state that combines:
    1. Meta: Request/Response lifecycle data
    2. Persona: Probabilistic social simulation (BayesACT)
    3. Logic: Deterministic rule execution
    4. Context: The raw narrative stream
    """
    meta: Meta
    persona_state: PersonaState
    logic_layer: LogicLayer
    context_stream: ContextStream

    model_config = ConfigDict(populate_by_name=True)

# Alias for ease of use
HNASP = HNASPState
