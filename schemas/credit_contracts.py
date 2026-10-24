from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class EvidenceStatus(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    EXTRACTED = "EXTRACTED"
    CHALLENGED = "CHALLENGED"
    AUDITED = "AUDITED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class SourceLocator(BaseModel):
    artifact_sha256: str = Field(..., description="SHA-256 hash of the canonical source document")
    page_number: Optional[int] = Field(None, ge=1)
    paragraph_index: Optional[int] = Field(None, ge=0)
    sec_accession_number: Optional[str] = None
    section_title: Optional[str] = None


class EvidenceReference(BaseModel):
    evidence_id: str
    status: EvidenceStatus = EvidenceStatus.UNVERIFIED
    locator: SourceLocator
    extracted_at_timestamp: float
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    extractor_model: str


class NegativeFinding(BaseModel):
    finding_id: str
    target_metric: str
    searched_scope: List[str]
    exhaustion_reason: str
    severity: str = "CRITICAL"
    is_omission_verified: bool = True

    @field_validator("finding_id")
    @classmethod
    def validate_finding_id_prefix(cls, v: str) -> str:
        if not v.startswith("NEG-"):
            raise ValueError("finding_id must start with 'NEG-' prefix")
        return v


class FinancialAssertion(BaseModel):
    assertion_id: str
    metric_name: str
    value: float
    unit: str
    reporting_period: str
    evidence_ref: EvidenceReference
    is_audited: bool = False


class AuthorizedStateTransitionRequest(BaseModel):
    request_id: str
    current_state: str
    target_state: str
    proposed_by_agent: str
    sanctioning_officer_id: Optional[str] = None
    is_authorized: bool = False
    justification: str

    @model_validator(mode="after")
    def validate_fiduciary_authorization(self) -> "AuthorizedStateTransitionRequest":
        if self.is_authorized and not self.sanctioning_officer_id:
            raise ValueError(
                "Fiduciary Authorization Failure: is_authorized cannot be True without a valid human sanctioning_officer_id"
            )
        return self
