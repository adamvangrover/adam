from __future__ import annotations
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import uuid

class ThoughtNode(BaseModel):
    """
    Represents a discrete unit of thought or reasoning.
    Merged: Uses 'thought_id' (main) for explicit naming, but retains the
    graph structure (parent/children) from the prompt-guide branch to allow backtracking.
    """
    thought_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    confidence: float = Field(default=0.0, description="Confidence score between 0.0 and 1.0")
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_child(self, child_id: str):
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

class StrategicPlan(BaseModel):
    """
    Represents a structured plan with specific goals and actionable steps.
    Merged: Combines 'goals' (main) for high-level intent with 'status' (guide) for state tracking.
    """
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goals: List[str] = Field(default_factory=list, description="High-level objectives")
    steps: List[str] = Field(default_factory=list, description="Actionable execution steps")
    status: str = "DRAFT"  # ACTIVE, COMPLETED, FAILED

class CognitiveState(BaseModel):
    """
    Represents the internal 'mind' of an agent.
    Merged: Retains the 'thought_graph' (guide) for complex history tracking,
    but provides properties to access active context easily.
    """
    current_thought_id: Optional[str] = None
    thought_graph: Dict[str, ThoughtNode] = Field(default_factory=dict)
    plans: List[StrategicPlan] = Field(default_factory=list)
    active_plan_id: Optional[str] = None

    @property
    def current_thought(self) -> Optional[ThoughtNode]:
        """Convenience property to get the actual thought object."""
        if self.current_thought_id and self.current_thought_id in self.thought_graph:
            return self.thought_graph[self.current_thought_id]
        return None

    @property
    def active_plan(self) -> Optional[StrategicPlan]:
        """Convenience property to retrieve the full active plan object."""
        if not self.active_plan_id:
            return None
        return next((p for p in self.plans if p.plan_id == self.active_plan_id), None)

    def add_thought(self, content: str, confidence: float = 0.0, parent_id: Optional[str] = None) -> ThoughtNode:
        """
        Creates a new thought, links it to a parent (if any), and stores it.
        """
        # Auto-link to current thought if no parent provided and a current thought exists
        if parent_id is None and self.current_thought_id:
            parent_id = self.current_thought_id

        thought = ThoughtNode(content=content, confidence=confidence, parent_id=parent_id)
        self.thought_graph[thought.thought_id] = thought

        if parent_id and parent_id in self.thought_graph:
            self.thought_graph[parent_id].add_child(thought.thought_id)

        self.current_thought_id = thought.thought_id
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

    def create_plan(self, goals: List[str], steps: List[str]) -> StrategicPlan:
        plan = StrategicPlan(goals=goals, steps=steps, status="ACTIVE")
        self.plans.append(plan)
        self.active_plan_id = plan.plan_id
        return plan