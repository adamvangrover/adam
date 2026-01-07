from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uuid

class ThoughtNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    score: float = 0.0
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)

    def add_child(self, child_id: str):
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

class StrategicPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[str] = Field(default_factory=list)
    status: str = "DRAFT" # ACTIVE, COMPLETED, FAILED

class CognitiveState(BaseModel):
    """
    Represents the internal 'mind' of an agent, tracking its thoughts and plans.
    """
    current_thought: Optional[ThoughtNode] = None
    thought_graph: Dict[str, ThoughtNode] = Field(default_factory=dict)
    plans: List[StrategicPlan] = Field(default_factory=list)
    active_plan_id: Optional[str] = None

    def add_thought(self, content: str, score: float = 0.0, parent_id: Optional[str] = None) -> ThoughtNode:
        """
        Creates a new thought, links it to a parent (if any), and stores it.
        """
        thought = ThoughtNode(content=content, score=score, parent_id=parent_id)
        self.thought_graph[thought.id] = thought

        if parent_id and parent_id in self.thought_graph:
            self.thought_graph[parent_id].add_child(thought.id)

        self.current_thought = thought
        return thought

    def get_thought_trace(self, thought_id: str) -> List[ThoughtNode]:
        """
        Backtracks from a given thought ID to the root to reconstruct a reasoning chain.
        """
        trace = []
        current_id = thought_id
        while current_id and current_id in self.thought_graph:
            node = self.thought_graph[current_id]
            trace.append(node)
            current_id = node.parent_id
        return list(reversed(trace))

    def create_plan(self, steps: List[str]) -> StrategicPlan:
        plan = StrategicPlan(steps=steps, status="ACTIVE")
        self.plans.append(plan)
        self.active_plan_id = plan.plan_id
        return plan
