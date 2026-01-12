from __future__ import annotations
from typing import List, Dict, Any, Optional, Annotated
import operator
from pydantic import BaseModel, Field
from core.schemas.v23_5_schema import DeepDiveRequest, ExecutionPlan, CritiqueResult

# Define reducer for message appending if using standard LangGraph messages
def merge_dicts(a: Dict, b: Dict) -> Dict:
    return {**a, **b}

class AgentState(BaseModel):
    """
    The shared state for the Cyclical Reasoning Graph.
    Uses Pydantic for validation, compatible with LangGraph.
    """
    # Request Context
    request: Optional[DeepDiveRequest] = None
    thread_id: str = Field(default="")

    # Planning
    plan: Optional[ExecutionPlan] = None

    # Execution & Reasoning
    draft: str = Field(default="")
    critique: Optional[CritiqueResult] = None
    revision_count: int = Field(default=0)

    # Data Context
    retrieved_data: List[Any] = Field(default_factory=list)
    tool_outputs: Dict[str, Any] = Field(default_factory=dict)

    # Flags
    is_complete: bool = False
    error: Optional[str] = None

# For LangGraph Functional API (TypedDict equivalent)
from typing import TypedDict

class GraphState(TypedDict):
    """
    TypedDict version for LangGraph compatibility (if Pydantic is not supported directly in the version used).
    """
    request: DeepDiveRequest
    plan: ExecutionPlan
    draft: str
    critique: CritiqueResult
    revision_count: int
    retrieved_data: List[Any]
    tool_outputs: Dict[str, Any]
