from pydantic import BaseModel, Field
from typing import List, Optional

class ThoughtNode(BaseModel):
    """
    Minimal placeholder for ThoughtNode.
    """
    thought_id: str
    content: str
    confidence: float

class StrategicPlan(BaseModel):
    """
    Minimal placeholder for StrategicPlan.
    """
    plan_id: str
    steps: List[str]

class CognitiveState(BaseModel):
    """
    Minimal placeholder for CognitiveState.
    """
    current_thought: Optional[ThoughtNode] = None
    active_plan: Optional[StrategicPlan] = None
