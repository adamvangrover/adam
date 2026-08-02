from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class DecisionNode(BaseModel):
    """A single evaluation node within the decision graph (e.g., Liquidity Check)."""
    node_id: str = Field(..., description="Unique identifier for the decision node")
    concept_name: str = Field(..., description="The risk concept being evaluated (e.g., 'Liquidity', 'Leverage')")
    result_value: Any = Field(..., description="The computed outcome for this node")
    evidence_refs: List[str] = Field(default_factory=list, description="IDs of evidence used in this calculation")
    policy_ref: Optional[str] = Field(None, description="The policy ID and version used to evaluate this node")

class DecisionEdge(BaseModel):
    """A directed edge indicating that one decision's output influenced another."""
    source_node_id: str
    target_node_id: str
    weight: Optional[float] = None

class DecisionGraph(BaseModel):
    """The fully explainable graph culminating in a final decision (e.g., Risk Rating)."""
    graph_id: str = Field(..., description="Unique identifier for this decision graph execution")
    nodes: Dict[str, DecisionNode] = Field(default_factory=dict, description="All intermediate decisions")
    edges: List[DecisionEdge] = Field(default_factory=list, description="Dependencies between decisions")
    final_decision_node_id: str = Field(..., description="The node containing the ultimate outcome (e.g., Final Rating)")
    provenance_hash: str = Field(..., description="W3C PROV-O compliant deterministic hash of the graph")

    def get_final_outcome(self) -> Any:
        return self.nodes[self.final_decision_node_id].result_value
