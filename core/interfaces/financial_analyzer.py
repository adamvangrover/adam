from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# --- Shared Data Models for Financial Analysis ---

class RiskFactor(BaseModel):
    category: str = Field(..., description="Risk category (e.g., Market, Operational, Regulatory)")
    description: str = Field(..., description="Detailed description of the risk factor")
    severity: str = Field(..., description="Severity level: Low, Medium, High, Critical")
    time_horizon: str = Field(..., description="Time horizon: Short-term, Medium-term, Long-term")
    mitigation_strategy: Optional[str] = Field(None, description="Potential mitigation strategies mentioned")

class StrategicInsight(BaseModel):
    theme: str = Field(..., description="Strategic theme or initiative")
    details: str = Field(..., description="Details of the strategy")
    confidence_score: float = Field(..., description="Confidence score (0.0 - 1.0) based on language certainty")
    sentiment: str = Field(..., description="Sentiment: Positive, Neutral, Negative")

class ESGMetric(BaseModel):
    category: str = Field(..., description="Environmental, Social, or Governance")
    score: str = Field(..., description="Qualitative score: Poor, Average, Good, Leading")
    evidence: str = Field(..., description="Quote or evidence supporting the score")

class CompetitorDynamic(BaseModel):
    competitor_name: str = Field(..., description="Name of the competitor")
    threat_level: str = Field(..., description="Threat level: Low, Moderate, High")
    market_positioning: str = Field(..., description="Analysis of relative market positioning")

class ForwardGuidance(BaseModel):
    metric: str = Field(..., description="Financial metric (e.g., Revenue, EPS)")
    prediction: str = Field(..., description="Management's prediction or target")
    timeframe: str = Field(..., description="Timeframe for the guidance")

# --- Deep Research Topics (v24.0 Expansion) ---

class SupplyChainNode(BaseModel):
    entity: str = Field(..., description="Supplier or Partner Name")
    role: str = Field(..., description="Role in supply chain (e.g., Chip Manufacturer, Logistics)")
    location: str = Field(..., description="Geographic location")
    risk_level: str = Field(..., description="Supply chain risk level: Low, Medium, High")

class GeopoliticalExposure(BaseModel):
    region: str = Field(..., description="Region (e.g., APAC, EMEA)")
    exposure_type: str = Field(..., description="Revenue, Manufacturing, or R&D")
    risk_description: str = Field(..., description="Specific geopolitical risks (e.g., Sanctions, Instability)")

class TechnologicalMoat(BaseModel):
    technology: str = Field(..., description="Core technology or IP")
    moat_strength: str = Field(..., description="Strength of moat: Narrow, Wide")
    sustainability: str = Field(..., description="How sustainable is this advantage over 3-5 years?")

class FinancialAnalysisResult(BaseModel):
    """
    Standardized output for all financial analyzers.
    Expanded for v24.0 to include Deep Research Topics.
    """
    ticker: str
    period: str
    risk_factors: List[RiskFactor]
    strategic_insights: List[StrategicInsight]
    esg_metrics: List[ESGMetric]
    competitor_dynamics: List[CompetitorDynamic]
    forward_guidance: List[ForwardGuidance]

    # v24.0 Additions
    supply_chain_map: List[SupplyChainNode] = Field(default_factory=list)
    geopolitical_exposure: List[GeopoliticalExposure] = Field(default_factory=list)
    technological_moat: List[TechnologicalMoat] = Field(default_factory=list)

    summary_narrative: str
    raw_source_metadata: Dict[str, Any] = Field(default_factory=dict)

# --- Base Interface ---

class BaseFinancialAnalyzer(ABC):
    """
    Standard interface for financial analysis engines.
    Allows swapping between different LLM providers (Gemini, OpenAI, etc.)
    or analysis methodologies.
    """

    @abstractmethod
    async def analyze_report(self, report_text: str, context: Dict[str, Any] = None) -> FinancialAnalysisResult:
        """
        Analyzes a financial report text and returns structured insights.

        Args:
            report_text: The raw text of the financial report (10-K, 10-Q, earnings call).
            context: Additional context like ticker, year, quarter, sector data.

        Returns:
            FinancialAnalysisResult: Validated Pydantic object containing the analysis.
        """
        pass

    @abstractmethod
    async def analyze_multimodal(self, image_paths: List[str], prompt: str) -> str:
        """
        Performs analysis on visual financial data (charts, graphs).
        """
        pass
