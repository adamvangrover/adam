from __future__ import annotations

from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Meta(BaseModel):
    version: str = "2.1.0"
    author: str
    timestamp: datetime = Field(default_factory=datetime.now)

class Config(BaseModel):
    mode: Literal["VC_SPONSOR", "CORP_FINANCE", "DISTRESSED"]
    calc_engine: Literal["ITERATIVE", "ANALYTIC"]
    regulatory_standard: str = "SNC_2025"

class ValuationContext(BaseModel):
    wacc_method: Literal["CAPM_STANDARD", "CAPM_SIZE_ADJUSTED", "BUILD_UP"]
    terminal_method: Literal["PERPETUITY_GROWTH", "EXIT_MULTIPLE", "DUAL_WEIGHTED"]
    growth_rate_perpetuity: float
    exit_multiple: float
    # Weights for Dual Method
    weight_perpetuity: Optional[float] = 0.5
    weight_exit_multiple: Optional[float] = 0.5

    # Cost of Equity Params
    risk_free_rate: float
    market_return: float
    beta: float
    size_premium: Optional[float] = 0.0
    specific_risk_premium: Optional[float] = 0.0

    # Cost of Debt Params
    pre_tax_cost_of_debt: float
    tax_rate: float

class Security(BaseModel):
    name: str
    security_type: Literal["REVOLVER", "TERM_LOAN", "MEZZANINE", "PREFERRED", "COMMON"]
    priority: int # 1 = Highest
    balance: float
    interest_rate: float
    is_pik: bool = False

    # For Preferred/Equity
    shares: Optional[float] = 0.0
    investment: Optional[float] = 0.0
    liq_pref_multiple: Optional[float] = 1.0
    is_participating: Optional[bool] = False
    conversion_ratio: Optional[float] = 1.0

    # Metadata
    sofr_spread: Optional[float] = None # For dynamic repricing

class CapitalStructure(BaseModel):
    securities: List[Security]

class CreditChallenge(BaseModel):
    stress_test_active: bool
    scenario: str
    pd_method: Literal["MERTON_STRUCTURAL", "LOGISTIC_HYBRID"]

    # Overrides for stress testing
    revenue_haircut: Optional[float] = 0.0
    margin_haircut: Optional[float] = 0.0
    multiple_contraction: Optional[float] = 0.0

    # PD Params
    sofr_base_rate: Optional[float] = 0.04
    asset_volatility: Optional[float] = 0.30
    default_point_leverage: Optional[float] = None # Optional override

class Collateral(BaseModel):
    cash_equivalents: float
    accounts_receivable: float
    inventory: float
    ppe: float
    intangibles: float

class Financials(BaseModel):
    """Simplified Financials for DCF"""
    historical_revenue: List[float]
    projected_revenue_growth: List[float]
    historical_ebitda_margin: List[float]
    projected_ebitda_margin: List[float]
    capex_percent_revenue: float
    nwc_percent_revenue: float
    depreciation_percent_revenue: float

    current_year_revenue: float

class WorkstreamContext(BaseModel):
    meta: Meta
    config: Config
    valuation_context: ValuationContext
    capital_structure: CapitalStructure
    credit_challenge: CreditChallenge
    financials: Financials
    collateral: Optional[Collateral] = None

    model_config = ConfigDict(populate_by_name=True)

    def clone(self) -> WorkstreamContext:
        """Creates a deep copy of the context for simulation/stress-testing."""
        return self.model_copy(deep=True)

    def set_override(self, path: str, value: Any):
        """
        Sets a value in the context by dot-notation path.
        Example: 'valuation_context.growth_rate_perpetuity'
        """
        keys = path.split('.')
        obj = self
        for key in keys[:-1]:
            obj = getattr(obj, key)
        setattr(obj, keys[-1], value)
