from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class SeverityScore(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ConformanceStatus(str, Enum):
    FULL_CONFORMANCE = "Full Conformance"
    CONFORMANCE_WITH_EXCEPTIONS = "Conformance with Exceptions"
    NON_CONFORMANT = "Non-Conformant"

class FindingStatus(str, Enum):
    CONFORMANT = "Conformant"
    NON_CONFORMANT = "Non-Conformant"
    AMBIGUITY = "Ambiguity"

class PolicyStandard(BaseModel):
    source: str
    clause: str
    text: str

class DocumentReference(BaseModel):
    source: str
    clause: str
    text: str

class VerificationQuestion(BaseModel):
    question: str
    answer: str

class VerificationTrail(BaseModel):
    verificationQuestions: List[VerificationQuestion]
    verificationOutcome: str

class Finding(BaseModel):
    status: FindingStatus
    severityScore: SeverityScore
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    remediationAction: str
    policyStandard: PolicyStandard
    documentReference: DocumentReference
    analysis: str
    verificationTrail: VerificationTrail

class ReportMetadata(BaseModel):
    documentReviewed: str
    documentID: str
    reviewDate: str
    reviewerPersona: str = "Credit Risk Control Officer"
    overallConformanceStatus: ConformanceStatus

class CreditConformanceReport(BaseModel):
    reportMetadata: ReportMetadata
    findings: List[Finding]

    model_config = ConfigDict(populate_by_name=True)
