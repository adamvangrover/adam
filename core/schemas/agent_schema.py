from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from src.schemas.core_types import AgentInput as CoreAgentInput, AgentOutput as CoreAgentOutput
from src.pdil.models import ProvenanceHeader

class AgentInput(BaseModel):
    query: str = Field(..., description="The specific question or objective.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Shared graph state (RAG data, previous results).")
    tools: List[str] = Field(default_factory=list, description="List of allowed tool names.")

class AgentOutput(CoreAgentOutput):
    answer: str = Field(..., description="The final synthesized answer.")
    sources: List[str] = Field(default_factory=list, description="List of citations (filenames, URLs).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Conviction score (0.0 to 1.0).")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Debug info, token usage, etc.")
    provenance_trace: ProvenanceHeader = Field(..., description="Immutable provenance trace linking the agent's output to source and logic version.")

class FundamentalReport(BaseModel):
    company_id: str = Field(..., description="The ticker or company ID.")
    enterprise_value: float = Field(..., description="The calculated Enterprise Value (EV).")
    wacc: float = Field(..., description="The Weighted Average Cost of Capital (WACC).")
    terminal_growth_rate: float = Field(..., description="The terminal growth rate applied.")
    financial_health: str = Field(..., description="The overall financial health assessment.")
    dcf_scenarios: Dict[str, float] = Field(..., description="The DCF valuations under Base, Bull, and Bear scenarios.")
    analysis_summary: str = Field(..., description="The complete summary analysis.")
