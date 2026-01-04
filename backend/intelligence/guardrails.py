from pydantic import BaseModel, Field, model_validator, ValidationInfo
from typing import Optional
from decimal import Decimal

class CreditDecision(BaseModel):
    """
    Deterministic credit decision model based on strict financial rules.
    Used for initial automated screening.
    """
    applicant_id: str
    annual_income: Decimal = Field(..., gt=0)
    total_debt: Decimal = Field(..., ge=0)
    requested_amount: Decimal = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=850)

    decision: Optional[str] = None
    reason: Optional[str] = None

    @model_validator(mode='after')
    def enforce_credit_policy(self) -> 'CreditDecision':
        # Calculate DTI
        monthly_income = self.annual_income / 12
        # Assuming total_debt is monthly debt service for simplicity, or we should clarify.
        # If total_debt is total outstanding, we need a monthly payment estimation.
        # Let's assume total_debt passed here is the total amount, and we estimate 3% monthly payment
        # OR assume the input is already monthly debt obligations.
        # "If Debt-to-Income > 40%, reject".

        # Let's interpret 'total_debt' as 'monthly_debt_obligations' for DTI calculation context,
        # or we calculate a mock monthly payment.
        # For this prototype, let's assume `total_debt` is the total debt load and we use a standard
        # amortization or simply check Debt/Income ratio on annual basis?
        # Usually DTI is monthly debt / monthly gross income.

        # Let's assume the field `total_debt` represents *monthly* debt payments for this guardrail model
        # to match standard DTI inputs, or rename it.
        # I'll stick to the user prompt "Debt-to-Income > 40%".
        # Let's assume input is annual data, so we compare Total Debt / Annual Income ?
        # No, DTI is usually monthly.
        # I will assume `total_debt` is total debt and I will estimate monthly payment as 1% of total debt (proxy).

        estimated_monthly_debt_payment = self.total_debt * Decimal("0.01") # Rough proxy
        dti = estimated_monthly_debt_payment / monthly_income

        if dti > Decimal("0.40"):
            self.decision = "REJECT"
            self.reason = f"DTI {dti:.2%} exceeds 40% limit."
            return self

        if self.credit_score < 600:
            self.decision = "REJECT"
            self.reason = "Credit score below 600."
            return self

        self.decision = "APPROVE"
        self.reason = "Meets all criteria."
        return self

class MarketRiskGuardrail(BaseModel):
    """
    Checks order parameters against risk limits.
    """
    symbol: str
    quantity: int
    price: Decimal
    max_order_value: Decimal = Decimal("1000000.00") # $1M limit

    @model_validator(mode='after')
    def check_order_limits(self) -> 'MarketRiskGuardrail':
        order_value = self.quantity * self.price
        if order_value > self.max_order_value:
            raise ValueError(f"Order value {order_value} exceeds limit {self.max_order_value}")
        return self
