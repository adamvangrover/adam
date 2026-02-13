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

class CreditMemo(BaseModel):
    """
    The complete Credit Memo document.
    """
    borrower_name: str
    report_date: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    executive_summary: str
    sections: List[CreditMemoSection] = []
    financial_ratios: Dict[str, float] = {}
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
