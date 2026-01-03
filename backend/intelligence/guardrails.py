from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Optional, List
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class CreditDecision(BaseModel):
    decision: str = Field(..., description="APPROVE or REJECT")
    risk_level: RiskLevel
    score: int = Field(..., ge=0, le=100)
    rationale: str

    @field_validator('decision')
    @classmethod
    def validate_decision(cls, v: str, info: ValidationInfo) -> str:
        if v not in ["APPROVE", "REJECT"]:
            raise ValueError("Decision must be APPROVE or REJECT")
        return v

class FinancialMetrics(BaseModel):
    debt_to_income: float
    credit_score: int
    loan_amount: float
    annual_income: float

    @field_validator('debt_to_income')
    @classmethod
    def check_dti(cls, v: float) -> float:
        if v > 0.43: # Standard mortgage limit
            # This logic might belong in the decision logic, but here as a guardrail warning
            pass
        return v

def evaluate_credit(metrics: FinancialMetrics) -> CreditDecision:
    """
    Deterministic evaluation based on Pydantic guardrails.
    """
    if metrics.debt_to_income > 0.40:
        return CreditDecision(
            decision="REJECT",
            risk_level=RiskLevel.HIGH,
            score=50,
            rationale="Debt-to-Income ratio exceeds 40% threshold."
        )

    if metrics.credit_score < 600:
        return CreditDecision(
            decision="REJECT",
            risk_level=RiskLevel.HIGH,
            score=metrics.credit_score // 10,
            rationale="Credit score below minimum threshold."
        )

    return CreditDecision(
        decision="APPROVE",
        risk_level=RiskLevel.LOW,
        score=90,
        rationale="Metrics within acceptable range."
    )
