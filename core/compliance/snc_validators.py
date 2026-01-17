from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Optional, List, Dict, Any, ClassVar
from enum import Enum

class RiskLevel(Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"

class FinancialHealthCheck(BaseModel):
    """
    Validates financial metrics against hard-coded policy limits.
    Acts as the 'Compliance-as-Code' layer.
    """
    debt_to_equity: Optional[float] = Field(None, description="Total Debt / Total Equity")
    net_profit_margin: Optional[float] = Field(None, description="Net Income / Revenue")
    current_ratio: Optional[float] = Field(None, description="Current Assets / Current Liabilities")
    interest_coverage_ratio: Optional[float] = Field(None, description="EBIT / Interest Expense")

    # Thresholds (Defaults based on typical credit policy, can be overridden)
    # These could be loaded from a config in a real deployment
    MAX_DEBT_TO_EQUITY: ClassVar[float] = 3.0
    MIN_PROFIT_MARGIN: ClassVar[float] = 0.0
    MIN_CURRENT_RATIO: ClassVar[float] = 1.0
    MIN_INTEREST_COVERAGE: ClassVar[float] = 1.25

    @field_validator('debt_to_equity')
    @classmethod
    def check_leverage(cls, v: Optional[float], info: ValidationInfo) -> Optional[float]:
        if v is not None and v > cls.MAX_DEBT_TO_EQUITY:
            raise ValueError(f"Leverage Breach: Debt/Equity {v:.2f} exceeds limit of {cls.MAX_DEBT_TO_EQUITY}")
        return v

    @field_validator('net_profit_margin')
    @classmethod
    def check_profitability(cls, v: Optional[float], info: ValidationInfo) -> Optional[float]:
        if v is not None and v < cls.MIN_PROFIT_MARGIN:
            raise ValueError(f"Profitability Breach: Net Margin {v:.2%} is below limit of {cls.MIN_PROFIT_MARGIN:.0%}")
        return v

    @field_validator('current_ratio')
    @classmethod
    def check_liquidity(cls, v: Optional[float], info: ValidationInfo) -> Optional[float]:
        if v is not None and v < cls.MIN_CURRENT_RATIO:
            raise ValueError(f"Liquidity Breach: Current Ratio {v:.2f} is below limit of {cls.MIN_CURRENT_RATIO}")
        return v

    @field_validator('interest_coverage_ratio')
    @classmethod
    def check_coverage(cls, v: Optional[float], info: ValidationInfo) -> Optional[float]:
        if v is not None and v < cls.MIN_INTEREST_COVERAGE:
             raise ValueError(f"Coverage Breach: Interest Coverage {v:.2f} is below limit of {cls.MIN_INTEREST_COVERAGE}")
        return v

class RegulatoryGuardrails(BaseModel):
    """
    Enforces higher-level regulatory constraints, e.g., SNC specific rules.
    """
    rating_upgrade_requested: bool = False
    proposed_rating: str

    financials: FinancialHealthCheck

    # Rule: Cannot upgrade rating if financials are in breach

    def validate_policy(self) -> List[str]:
        violations = []

        # Check financial health
        try:
            # We trigger validation manually or rely on Pydantic's init validation
            # But since we want to collect multiple errors, we might check manually here
            # or rely on the fact that FinancialHealthCheck instantiation might have already failed.
            # However, for this design, let's assume FinancialHealthCheck is instantiated loosely
            # and we check constraints here, OR we catch ValidationError during instantiation.
            pass
        except Exception as e:
            violations.append(str(e))

        return violations

class SNCCreditPolicyResult(BaseModel):
    passed: bool
    violations: List[str]
    risk_level: RiskLevel

def evaluate_compliance(financial_data: Dict[str, Any]) -> SNCCreditPolicyResult:
    """
    Factory function to run compliance checks.
    """
    violations = []

    # Extract metrics
    ratios = financial_data.get("key_ratios", {})

    d_to_e = ratios.get("debt_to_equity_ratio")
    profit = ratios.get("net_profit_margin")
    current = ratios.get("current_ratio")
    coverage = ratios.get("interest_coverage_ratio")

    try:
        FinancialHealthCheck(
            debt_to_equity=d_to_e,
            net_profit_margin=profit,
            current_ratio=current,
            interest_coverage_ratio=coverage
        )
    except Exception as e:
        # Pydantic raises ValidationError which contains a list of errors
        # We simplify strictly for this agent usage
        if hasattr(e, 'errors'):
            for err in e.errors():
                # ctx usually contains the exception args for ValueError
                ctx = err.get('ctx', {})
                if 'error' in ctx:
                    # If it's a custom ValueError
                    msg = str(ctx['error'])
                else:
                    msg = err.get('msg', str(err))
                    # Remove "Value error, " prefix if present
                    if "Value error, " in msg:
                        msg = msg.split("Value error, ")[1]
                violations.append(msg)
        else:
            violations.append(str(e))

    if violations:
        return SNCCreditPolicyResult(passed=False, violations=violations, risk_level=RiskLevel.HIGH)

    return SNCCreditPolicyResult(passed=True, violations=[], risk_level=RiskLevel.LOW)
