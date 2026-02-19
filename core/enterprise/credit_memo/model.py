from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

class EvidenceChunk(BaseModel):
    """
    Represents a chunk of text from a source document with spatial metadata.
    Protocol: Spatial RAG
    """
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    text: str
    page_number: int
    bbox: List[float]  # [x0, y0, x1, y1] normalized coordinates (0-1)
    confidence: float = 1.0
    chunk_type: str = "Unstructured" # e.g., "Credit Agreement", "10-K", "Financial Table"

class Citation(BaseModel):
    """
    A citation linking a claim to a specific evidence chunk.
    """
    doc_id: str
    chunk_id: str
    page_number: int

class CreditMemoSection(BaseModel):
    title: str
    content: str
    citations: List[Citation] = []
    author_agent: str = "Writer" # e.g., "Risk Officer", "Quant", "Archivist"

class DCFAnalysis(BaseModel):
    """
    Deterministic Discounted Cash Flow Analysis.
    """
    free_cash_flow: List[float] # Projected FCF for 5 years
    growth_rate: float
    wacc: float
    terminal_value: float
    enterprise_value: float
    equity_value: float
    share_price: float
    inputs: Dict[str, float] = {} # Interactive inputs for frontend customization

class FinancialSpread(BaseModel):
    """
    Standardized financial template (FIBO-aligned).
    """
    total_assets: float
    total_liabilities: float
    total_equity: float
    revenue: float
    ebitda: float
    net_income: float
    interest_expense: float
    dscr: float
    leverage_ratio: float
    current_ratio: float
    period: str
    # Extended Fields
    gross_debt: Optional[float] = 0.0
    cash: Optional[float] = 0.0
    capex: Optional[float] = 0.0
    fcf: Optional[float] = 0.0 # Free Cash Flow
    ebitda_adjustments: Optional[Dict[str, float]] = {}

class CreditRating(BaseModel):
    agency: str
    rating: str
    outlook: str
    date: str

class DebtFacility(BaseModel):
    facility_type: str
    amount_committed: float
    amount_drawn: float
    interest_rate: str
    maturity_date: str
    snc_rating: str = "Pass"
    drc: float = 1.0 # 0-100%
    ltv: float = 0.5 # 0-100%
    conviction_score: float = 0.9 # 0-100%
    lgd: float = 0.4 # Loss Given Default (0-1)
    recovery_rate: float = 0.6 # Recovery Rate (0-1)

class EquityMarketData(BaseModel):
    market_cap: float
    share_price: float
    volume_avg_30d: float
    pe_ratio: float
    dividend_yield: float
    beta: float

class PeerComp(BaseModel):
    ticker: str
    name: str
    ev_ebitda: float
    pe_ratio: float
    leverage_ratio: float
    market_cap: float

class PDModel(BaseModel):
    """
    Probability of Default Model outputs.
    """
    input_factors: Dict[str, float] # e.g., "Leverage": 0.3, "Coverage": 0.4
    model_score: float # 0-100
    implied_rating: str
    one_year_pd: float
    five_year_pd: float

class LGDAnalysis(BaseModel):
    """
    Loss Given Default Analysis.
    """
    seniority_structure: List[Dict[str, Any]] # Tranches with recovery estimates
    recovery_rate_assumption: float
    loss_given_default: float

class Scenario(BaseModel):
    name: str
    probability: float
    revenue_growth: float
    ebitda_margin: float
    implied_share_price: float

class ScenarioAnalysis(BaseModel):
    scenarios: List[Scenario]
    weighted_share_price: float

class SystemTwoCritique(BaseModel):
    """
    System 2 Critique points and conviction score.
    """
    critique_points: List[str]
    conviction_score: float
    verification_status: str
    author_agent: str = "System 2"
    quantitative_analysis: Optional[Dict[str, Any]] = {}
    qualitative_analysis: Optional[Dict[str, Any]] = {}

class AuditLogEntry(BaseModel):
    """
    A rigorous audit log entry for compliance.
    Protocol: Audit-as-Code
    """
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    user_id: str
    action: str
    model_version: str
    prompt_version: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    citations_count: int
    validation_status: str  # PASS/FAIL/WARNING
    validation_errors: List[str] = []

class RepaymentScheduleItem(BaseModel):
    fiscal_year: int
    total_principal: float
    total_interest: float
    facilities_breakdown: Dict[str, float]

class CreditMemo(BaseModel):
    """
    The complete Credit Memo document.
    """
    borrower_name: str
    report_date: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    executive_summary: str
    sections: List[CreditMemoSection] = []
    key_strengths: List[str] = []
    key_weaknesses: List[str] = []
    mitigants: List[str] = []
    financial_ratios: Dict[str, float] = {}
    historical_financials: List[Dict[str, Any]] = [] # List of FinancialSpread dictionaries
    financial_forecast: List[Dict[str, Any]] = [] # Projected FinancialSpread dictionaries
    debt_repayment_forecast: List[RepaymentScheduleItem] = []
    dcf_analysis: Optional[DCFAnalysis] = None
    pd_model: Optional[PDModel] = None
    lgd_analysis: Optional[LGDAnalysis] = None
    scenario_analysis: Optional[ScenarioAnalysis] = None
    system_two_critique: Optional[SystemTwoCritique] = None
    risk_score: float
    credit_ratings: List[CreditRating] = []
    debt_facilities: List[DebtFacility] = []
    equity_data: Optional[EquityMarketData] = None
    agent_log: List[AuditLogEntry] = []
    peer_comps: List[PeerComp] = []
