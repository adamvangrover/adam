from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class OntologyNode(BaseModel):
    """
    Represents an entity or concept in the Knowledge Graph.
    """
    id: str = Field(..., description="Unique identifier for the node.")
    node_type: str = Field(..., description="The type of the node (e.g., Company, Debt, Sector).")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties of the node.")

class DebtHierarchy(BaseModel):
    """
    Represents the hierarchical structure of a company's debt obligations.
    """
    company_id: str = Field(..., description="The ID of the company.")
    senior_debt: List[str] = Field(default_factory=list, description="List of senior debt IDs.")
    subordinated_debt: List[str] = Field(default_factory=list, description="List of subordinated debt IDs.")
    mezzanine_debt: List[str] = Field(default_factory=list, description="List of mezzanine debt IDs.")

class CrossDefaultLinkage(BaseModel):
    """
    Represents a link between two debt instruments where a default in one triggers a default in the other.
    """
    source_debt_id: str = Field(..., description="The ID of the debt that triggers the default.")
    target_debt_id: str = Field(..., description="The ID of the debt that is triggered to default.")
    trigger_condition: str = Field(..., description="A description or rule defining the condition for the cross-default.")
