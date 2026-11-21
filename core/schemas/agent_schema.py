"""Single source of truth for System 2 agent I/O (`AgentInput` / `AgentOutput`)."""
from typing import Any, Dict, List
from pydantic import BaseModel, Field, field_validator

from src.pdil.models import ProvenanceHeader

class AgentInput(BaseModel):
    """
    Standard input schema for all System 2 agents.
    """
    query: str = Field(..., description="The specific question or objective.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Shared graph state (RAG data, previous results).")
    tools: List[str] = Field(default_factory=list, description="List of allowed tool names.")


class AgentOutput(BaseModel):
    """
    Standard output schema for all System 2 agents.
    """
    answer: str = Field(..., description="The final synthesized answer.")
    sources: List[str] = Field(default_factory=list, description="List of citations (filenames, URLs).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Conviction score (0.0 to 1.0).")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Debug info, token usage, etc.")
    provenance_trace: List[ProvenanceHeader] = Field(default_factory=list, description="PROV-O headers linking the output to source data and logic version (empty = ungrounded).")
    observed_drift: bool = Field(default=False, description="Flag indicating if behavior or calculation drift was observed.")

    @field_validator("provenance_trace", mode="before")
    @classmethod
    def _wrap_single_header(cls, value: Any) -> Any:
        if isinstance(value, (ProvenanceHeader, dict)):
            return [value]
        return value

    def check_grounding(self) -> bool:
        """True iff there is at least one PROV-O header and every header names its source data object."""
        return bool(self.provenance_trace) and all(h.source_data_object for h in self.provenance_trace)

class FundamentalReport(BaseModel):
    """
    Pydantic schema for the output of the FundamentalAnalyst agent.
    """
    company_id: str = Field(..., description="The ticker or company ID.")
    enterprise_value: float = Field(..., description="The calculated Enterprise Value (EV).")
    wacc: float = Field(..., description="The Weighted Average Cost of Capital (WACC).")
    terminal_growth_rate: float = Field(..., description="The terminal growth rate applied.")
    financial_health: str = Field(..., description="The overall financial health assessment.")
    dcf_scenarios: Dict[str, float] = Field(..., description="The DCF valuations under Base, Bull, and Bear scenarios.")
    analysis_summary: str = Field(..., description="The complete summary analysis.")
