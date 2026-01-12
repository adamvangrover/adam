from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator

class FinancialMetric(BaseModel):
    """
    Strict schema for financial metrics to prevent hallucinated types.
    """
    metric_name: str
    value: float
    unit: Literal['USD', 'EUR', 'JPY', 'CNY', 'GBP', 'Ratio', 'Percent']
    period: str = Field(description="Time period (e.g., '2023-Q4', 'TTM')")
    confidence: float = Field(ge=0.0, le=1.0, description="Conviction score")

    @field_validator('value')
    def check_value_sanity(cls, v):
        # Basic sanity check (e.g., no negative revenue unless specified)
        # For now, we just ensure it's a float, which Pydantic does automatically.
        return v

class Claim(BaseModel):
    """
    Represents a specific claim extracted from a document.
    """
    text: str
    source_id: str
    entity: str
    conviction_score: Optional[float] = None
    verification_method: Optional[str] = None

class ProvenanceMetadata(BaseModel):
    """
    W3C PROV-O compliant metadata.
    """
    source_id: str
    extraction_agent: str
    timestamp: str
    verification_method: str

class RiskFactor(BaseModel):
    category: Literal['Liquidity', 'Credit', 'Market', 'Operational', 'Geopolitical']
    severity: Literal['Low', 'Medium', 'High', 'Critical']
    description: str
    impact_probability: float = Field(ge=0.0, le=1.0)
    provenance: ProvenanceMetadata

class AnalysisReport(BaseModel):
    """
    The final output schema for a production-grade analysis.
    """
    target_entity: str
    report_type: Literal['Deep Dive', 'Risk Alert', 'Market Update']
    metrics: List[FinancialMetric]
    risks: List[RiskFactor]
    summary: str
    generated_at: str
