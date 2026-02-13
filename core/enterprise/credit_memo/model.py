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

class CreditMemo(BaseModel):
    """
    The complete Credit Memo document.
    """
    borrower_name: str
    report_date: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    executive_summary: str
    sections: List[CreditMemoSection] = []
    financial_ratios: Dict[str, float] = {}
    historical_financials: List[Dict[str, Any]] = [] # List of FinancialSpread dictionaries
    dcf_analysis: Optional[DCFAnalysis] = None
    risk_score: float

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
