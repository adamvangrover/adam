from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ThoughtNode(BaseModel):
    id: str
    content: str
    score: float = 0.0

class StrategicPlan(BaseModel):
    goals: List[str]
    steps: List[str]

class CognitiveState(BaseModel):
    current_thoughts: List[ThoughtNode] = Field(default_factory=list)
    plan: Optional[StrategicPlan] = None
