from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class SupplyChainDependency(BaseModel):
    """
    Represents a critical supply chain dependency for a company.
    """
    company_id: str = Field(..., description="The ID of the company.")
    supplier_id: str = Field(..., description="The ID of the critical supplier.")
    criticality_score: float = Field(..., ge=0.0, le=1.0, description="A score representing the criticality of the supplier.")
    dependency_type: str = Field(..., description="The type of dependency (e.g., Raw Materials, Component).")

class DistressTrigger(BaseModel):
    """
    Represents an early-warning distress trigger or event.
    """
    trigger_id: str = Field(..., description="Unique identifier for the trigger.")
    trigger_type: str = Field(..., description="The type of the trigger (e.g., Covenant Breach, Rating Downgrade).")
    severity: str = Field(..., description="The severity level of the trigger (e.g., High, Medium, Low).")
    description: str = Field(..., description="A detailed description of the event.")
    affected_entities: List[str] = Field(default_factory=list, description="List of IDs for affected entities.")

class SectorOracleInput(BaseModel):
    """
    Input schema for Sector-Specific Oracle agents.
    """
    sector: str = Field(..., description="The targeted sector (e.g., TMT, Healthcare).")
    monitoring_parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters to monitor for distress triggers.")
    target_entities: List[str] = Field(default_factory=list, description="Specific entities to focus the monitoring on.")
