from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo
from typing import Optional
from decimal import Decimal
from enum import Enum

# -----------------------------------------------------------------------------
# ENUMS & CONSTANTS
# -----------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class DecisionStatus(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    MANUAL_REVIEW = "MANUAL_REVIEW"

# -----------------------------------------------------------------------------
# DOMAIN MODELS (INPUTS)
# -----------------------------------------------------------------------------

class CreditApplication(BaseModel):
    """
    Standardized application payload for risk assessment.
    Uses Decimal for all monetary fields to ensure precision.
    """
    applicant_id: str
    annual_income: Decimal = Field(..., gt=0, description="Gross Annual Income")
    total_monthly_debt: Decimal = Field(..., ge=0, description="Total monthly debt obligations (rent, loans, etc)")
    requested_amount: Decimal = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=850)

    @property
    def debt_to_income_ratio(self) -> Decimal:
        """Calculates DTI based on monthly income."""
        monthly_income = self.annual_income / Decimal("12.0")
        if monthly_income == 0:
            return Decimal("Infinity")
        return self.total_monthly_debt / monthly_income

# -----------------------------------------------------------------------------
# DECISION MODELS (OUTPUTS)
# -----------------------------------------------------------------------------

class CreditDecision(BaseModel):
    """
    The deterministic output of the guardrail system.
    """
    decision: DecisionStatus
    risk_level: RiskLevel
    score: int = Field(..., ge=0, le=100, description="Internal proprietary risk score")
    rationale: str
    dti_calculated: Decimal

# -----------------------------------------------------------------------------
# GUARDRAIL LOGIC
# -----------------------------------------------------------------------------

class MarketRiskGuardrail(BaseModel):
    """
    Pre-trade check to ensure orders do not exceed capital limits.
    """
    symbol: str
    quantity: int = Field(..., gt=0)
    price: Decimal = Field(..., gt=0)
    max_order_value: Decimal = Decimal("1_000_000.00")  # $1M hard limit

    @model_validator(mode='after')
    def check_exposure_limits(self) -> 'MarketRiskGuardrail':
        order_value = self.quantity * self.price
        
        # Guardrail: Hard Exposure Limit
        if order_value > self.max_order_value:
            raise ValueError(
                f"FATAL: Order value ${order_value:,.2f} exceeds risk limit of ${self.max_order_value:,.2f}"
            )
        return self

def evaluate_credit_application(app: CreditApplication) -> CreditDecision:
    """
    Executes the credit policy guardrails. 
    Returns a deterministic decision based on DTI and FICO scores.
    """
    dti = app.debt_to_income_ratio

    # 1. Check DTI Guardrail (>43% is typically high risk)
    if dti > Decimal("0.43"):
        return CreditDecision(
            decision=DecisionStatus.REJECT,
            risk_level=RiskLevel.HIGH,
            score=40,
            rationale=f"DTI {dti:.2%} exceeds 43% tolerance.",
            dti_calculated=dti
        )

    # 2. Check Credit Score Guardrail
    if app.credit_score < 620:
        return CreditDecision(
            decision=DecisionStatus.REJECT,
            risk_level=RiskLevel.HIGH,
            score=50,
            rationale=f"Credit score {app.credit_score} below subprime threshold (620).",
            dti_calculated=dti
        )

    # 3. Check "Manual Review" Zone (620-680)
    if 620 <= app.credit_score < 680:
        return CreditDecision(
            decision=DecisionStatus.MANUAL_REVIEW,
            risk_level=RiskLevel.MEDIUM,
            score=70,
            rationale="Marginal credit profile requires underwriter review.",
            dti_calculated=dti
        )

    # 4. Approval
    return CreditDecision(
        decision=DecisionStatus.APPROVE,
        risk_level=RiskLevel.LOW,
        score=90,
        rationale="Automated approval: Strong credit and healthy DTI.",
        dti_calculated=dti
    )