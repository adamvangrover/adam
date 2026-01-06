from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ThoughtNode(BaseModel):
    """
    Represents a discrete unit of thought or reasoning.
    Merged: Uses 'thought_id' (main) and 'confidence' (main) but defaults to 0.0 (v24).
    """
    thought_id: str
    content: str
    confidence: float = Field(default=0.0, description="Confidence score between 0.0 and 1.0")

class StrategicPlan(BaseModel):
    """
    Represents a structured plan with specific goals and actionable steps.
    Merged: Retains 'goals' from v24 for context, and 'plan_id' from main for identification.
    """
    plan_id: str
    goals: List[str] = Field(default_factory=list, description="High-level objectives")
    steps: List[str] = Field(default_factory=list, description="Actionable execution steps")

class CognitiveState(BaseModel):
    """
    Represents the current mental snapshot of the agent.
    Merged: Uses v24's List structure for thoughts (richer context) 
    and main's explicit naming for 'active_plan'.
    """
    current_thoughts: List[ThoughtNode] = Field(default_factory=list)
    active_plan: Optional[StrategicPlan] = None