from pydantic import BaseModel, Field, field_validator, ValidationInfo, model_validator
from typing import Optional, List, Dict, Any, ClassVar
from enum import Enum

class RiskLevel(Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"

class SectorType(Enum):
    GENERAL = "General"
    ENERGY = "Energy"
    TECHNOLOGY = "Technology"
    REAL_ESTATE = "Real Estate"
    FINANCIALS = "Financials"

class FinancialHealthCheck(BaseModel):
    """
    Validates financial metrics against hard-coded policy limits.
    Acts as the 'Compliance-as-Code' layer.
    """
    sector: SectorType = Field(default=SectorType.GENERAL, description="Industry sector for specific guardrails")
    debt_to_equity: Optional[float] = Field(None, description="Total Debt / Total Equity")
    net_profit_margin: Optional[float] = Field(None, description="Net Income / Revenue")
    current_ratio: Optional[float] = Field(None, description="Current Assets / Current Liabilities")
    interest_coverage_ratio: Optional[float] = Field(None, description="EBIT / Interest Expense")

    # Base Thresholds (General)
    MAX_DEBT_TO_EQUITY: ClassVar[float] = 3.0
    MIN_PROFIT_MARGIN: ClassVar[float] = 0.0
    MIN_CURRENT_RATIO: ClassVar[float] = 1.0
    MIN_INTEREST_COVERAGE: ClassVar[float] = 1.25

    # Sector Specific Overrides
    SECTOR_LIMITS: ClassVar[Dict[SectorType, Dict[str, float]]] = {
        SectorType.ENERGY: {
            "MAX_DEBT_TO_EQUITY": 2.5,  # Stricter due to commodity volatility
            "MIN_CURRENT_RATIO": 1.2
        },
        SectorType.TECHNOLOGY: {
            "MAX_DEBT_TO_EQUITY": 4.0,  # Tolerates higher growth debt if cash rich (simplified)
            "MIN_PROFIT_MARGIN": -0.10  # Tolerates some burn
        },
        SectorType.REAL_ESTATE: {
            "MAX_DEBT_TO_EQUITY": 5.0,  # High leverage is standard
            "MIN_INTEREST_COVERAGE": 1.1
        }
    }

    @model_validator(mode='after')
    def check_sector_limits(self) -> 'FinancialHealthCheck':
        sector = self.sector
        limits = self.SECTOR_LIMITS.get(sector, {})

        # Determine effective limits
        max_de = limits.get("MAX_DEBT_TO_EQUITY", self.MAX_DEBT_TO_EQUITY)
        min_pm = limits.get("MIN_PROFIT_MARGIN", self.MIN_PROFIT_MARGIN)
        min_cr = limits.get("MIN_CURRENT_RATIO", self.MIN_CURRENT_RATIO)
        min_ic = limits.get("MIN_INTEREST_COVERAGE", self.MIN_INTEREST_COVERAGE)

        if self.debt_to_equity is not None and self.debt_to_equity > max_de:
            raise ValueError(f"Leverage Breach ({sector.value}): Debt/Equity {self.debt_to_equity:.2f} exceeds limit of {max_de}")

        if self.net_profit_margin is not None and self.net_profit_margin < min_pm:
             raise ValueError(f"Profitability Breach ({sector.value}): Net Margin {self.net_profit_margin:.2%} is below limit of {min_pm:.0%}")

        if self.current_ratio is not None and self.current_ratio < min_cr:
            raise ValueError(f"Liquidity Breach ({sector.value}): Current Ratio {self.current_ratio:.2f} is below limit of {min_cr}")

        if self.interest_coverage_ratio is not None and self.interest_coverage_ratio < min_ic:
             raise ValueError(f"Coverage Breach ({sector.value}): Interest Coverage {self.interest_coverage_ratio:.2f} is below limit of {min_ic}")

        return self

class StressTestBuffers(BaseModel):
    """
    Enforces capital buffers based on market volatility (VIX).
    """
    volatility_index: float = Field(..., description="Current VIX level")
    tier_1_capital_ratio: Optional[float] = Field(None, description="Tier 1 Capital / Risk-Weighted Assets")

    @model_validator(mode='after')
    def check_volatility_buffer(self) -> 'StressTestBuffers':
        vix = self.volatility_index
        t1 = self.tier_1_capital_ratio

        if t1 is None:
            return self

        required_buffer = 0.06 # Basel III min 6%

        if vix > 30:
            required_buffer = 0.08 # Require 8% in high stress
        elif vix > 20:
            required_buffer = 0.07 # Require 7% in moderate stress

        if t1 < required_buffer:
            raise ValueError(f"Stress Buffer Breach: VIX at {vix:.0f} requires Tier 1 Capital >= {required_buffer:.1%}, found {t1:.1%}")

        return self

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
        # Validation happens at model instantiation mostly
        return violations

class SNCCreditPolicyResult(BaseModel):
    passed: bool
    violations: List[str]
    risk_level: RiskLevel

def evaluate_compliance(financial_data: Dict[str, Any], sector_name: str = "General", market_data: Dict[str, Any] = None) -> SNCCreditPolicyResult:
    """
    Factory function to run compliance checks.
    """
    violations = []

    # Map string sector to Enum
    try:
        sector_enum = SectorType(sector_name)
    except ValueError:
        sector_enum = SectorType.GENERAL

    # Extract metrics
    ratios = financial_data.get("key_ratios", {})

    d_to_e = ratios.get("debt_to_equity_ratio")
    profit = ratios.get("net_profit_margin")
    current = ratios.get("current_ratio")
    coverage = ratios.get("interest_coverage_ratio")

    # 1. Check Financial Health
    try:
        FinancialHealthCheck(
            sector=sector_enum,
            debt_to_equity=d_to_e,
            net_profit_margin=profit,
            current_ratio=current,
            interest_coverage_ratio=coverage
        )
    except Exception as e:
        if hasattr(e, 'errors'):
            for err in e.errors():
                ctx = err.get('ctx', {})
                if 'error' in ctx:
                    msg = str(ctx['error'])
                else:
                    msg = err.get('msg', str(err))
                    if "Value error, " in msg:
                        msg = msg.split("Value error, ")[1]
                violations.append(msg)
        else:
            violations.append(str(e))

    # 2. Check Stress Buffers (if data available)
    if market_data:
        vix = market_data.get("vix")
        t1_ratio = ratios.get("tier_1_capital_ratio")

        if vix is not None and t1_ratio is not None:
            try:
                StressTestBuffers(
                    volatility_index=float(vix),
                    tier_1_capital_ratio=float(t1_ratio)
                )
            except Exception as e:
                 if hasattr(e, 'errors'):
                    for err in e.errors():
                        ctx = err.get('ctx', {})
                        if 'error' in ctx:
                            msg = str(ctx['error'])
                        else:
                            msg = err.get('msg', str(err))
                            if "Value error, " in msg:
                                msg = msg.split("Value error, ")[1]
                        violations.append(msg)
                 else:
                    violations.append(str(e))

    if violations:
        return SNCCreditPolicyResult(passed=False, violations=violations, risk_level=RiskLevel.HIGH)

    return SNCCreditPolicyResult(passed=True, violations=[], risk_level=RiskLevel.LOW)


class RatingUpgradeGuardrail(BaseModel):
    """
    Prevents rating upgrades if policy limits are breached.
    """
    current_rating: str
    proposed_rating: str
    compliance_result: SNCCreditPolicyResult

    @model_validator(mode='after')
    def prevent_unjustified_upgrade(self) -> 'RatingUpgradeGuardrail':
        # Simple logic: Rating improvement blocked if compliance failed
        # Assumption: If they are different, it's a change.
        # If compliance failed, ANY change is blocked to be safe (conservative policy).
        if self.current_rating != self.proposed_rating and not self.compliance_result.passed:
             # Remove "Value error, " prefix if present in violations for cleaner message
             cleaned_violations = [v.replace("Value error, ", "") for v in self.compliance_result.violations]
             raise ValueError(f"Compliance Block: Cannot change rating from {self.current_rating} to {self.proposed_rating} while policy violations exist: {cleaned_violations}")
        return self


def validate_deal(financial_data: Dict[str, Any], sector_name: str, market_data: Dict[str, Any] = None, current_rating: str = "NR", proposed_rating: str = "NR") -> SNCCreditPolicyResult:
    """
    Comprehensive validation wrapper combining financial health, stress buffers, and rating logic.
    """
    # 1. Evaluate Financial & Market Compliance
    result = evaluate_compliance(financial_data, sector_name, market_data)

    # 2. Check Rating Logic
    try:
        RatingUpgradeGuardrail(
            current_rating=current_rating,
            proposed_rating=proposed_rating,
            compliance_result=result
        )
    except Exception as e:
        # Append logic violation
        msg = str(e)
        if "Value error, " in msg:
            msg = msg.split("Value error, ")[1]

        result.passed = False
        result.violations.append(msg)
        result.risk_level = RiskLevel.CRITICAL

    return result
