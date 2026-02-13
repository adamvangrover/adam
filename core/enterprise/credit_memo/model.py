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

class Attribution(BaseModel):
    """
    Represents the attribution for a specific score or decision.
    Protocol: Explainable AI (XAI)
    """
    agent_id: str
    model_version: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    justification: str
    key_factors: List[str] = []

class Citation(BaseModel):
    """
    A citation linking a claim to a specific evidence chunk.
    """
    doc_id: str
    chunk_id: str
    page_number: int
    bbox: Optional[List[float]] = None
    text_snippet: Optional[str] = None

class CreditMemoSection(BaseModel):
    title: str
    content: str
    citations: List[Citation] = []

class DebtFacilityRating(BaseModel):
    """
    Represents a credit rating for a specific debt facility.
    """
    facility_type: str # e.g., "Senior Secured Term Loan B", "Senior Unsecured Notes"
    rating: str # e.g., "BBB-"
    recovery_rating: Optional[str] = None # e.g., "RR1" (90-100%)
    amount_outstanding: Optional[float] = None # In Millions

class CreditMemo(BaseModel):
    """
    The complete Credit Memo document.
    """
    borrower_name: str
    report_date: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    executive_summary: str
    sections: List[CreditMemoSection] = []
    financial_ratios: Dict[str, float] = {}

    # New Fields
    risk_score: float
    sentiment_score: float = Field(default=50.0)
    conviction_score: float = Field(default=50.0)
    credit_rating: str = "NR"

    # Valuation
    price_target: Optional[float] = None
    price_level: Optional[str] = None # "Undervalued", "Fair Value", "Overvalued"
    market_cap: Optional[float] = None # Total Market Value (Equity Value)
    enterprise_value: Optional[float] = None

    # Debt Structure
    debt_ratings: List[DebtFacilityRating] = []

    system_two_notes: Optional[str] = None # System Two output

    # XAI Attribution Map (Metric Name -> Attribution)
    score_attributions: Dict[str, Attribution] = {}

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

    # New: Full Reasoning Trace
    attribution_trace: Dict[str, Any] = {}
