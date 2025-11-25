from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field, ConfigDict

class RiskEntity(BaseModel):
    """
    Represents a single risk entity in the portfolio, based on the enhanced schema
    for kinetic risk simulation.
    """
    model_config = ConfigDict(populate_by_name=True)

    risk_id: str = Field(..., alias="Risk_ID", description="Unique alphanumeric identifier (e.g., R-CYB-004).")
    description: str = Field(..., description="A detailed description of the risk.")
    velocity: str = Field(..., description="Speed of onset: Instant, Rapid, Gradual, Latent.")
    persistence: str = Field(..., description="Duration of impact: Transient, Persistent, Permanent.")
    interconnectivity: List[str] = Field(..., description="Array of other Risk IDs triggered by this node.")
    strategic_objective: str = Field(..., description="The corporate goal threatened by this risk.")
    quantitative_exposure: float = Field(..., description="Financial Value at Risk (VaR) or Single Loss Expectancy (SLE).")
    control_effectiveness: float = Field(..., description="0.0 to 1.0 score of mitigation reliability.")
    control_strength: str = Field(..., description="Qualitative assessment of control reliability: Strong, Moderate, Weak.")

class CrisisSimulationInput(BaseModel):
    """
    Input schema for the Crisis Simulation prompt. It encapsulates all the dynamic
    data needed to run a simulation.
    """
    model_config = ConfigDict(populate_by_name=True)

    risk_portfolio: List[RiskEntity]
    current_date: str
    user_scenario: str

class CrisisLogEntry(BaseModel):
    """
    Represents a single entry in the chronological crisis simulation log.
    """
    model_config = ConfigDict(populate_by_name=True)

    timestamp: str = Field(..., description="The simulation time of the event (e.g., T+0, T+24:00).")
    event_description: str = Field(..., description="A narrative description of what occurred.")
    risk_id_cited: str = Field(..., description="The specific Risk ID that was triggered.")
    status: str = Field(..., description="The current status of the risk (e.g., Active, Mitigated, Escalating).")

class CrisisSimulationOutput(BaseModel):
    """
    Defines the structured output expected from the Crisis Simulation LLM call.
    """
    model_config = ConfigDict(populate_by_name=True)

    executive_summary: str = Field(..., description="High-level impact, total cost, and strategic implications.")
    crisis_simulation_log: List[CrisisLogEntry] = Field(..., description="A chronological timeline of simulated events.")
    recommendations: str = Field(..., description="Immediate, actionable mitigations based on the simulation.")
