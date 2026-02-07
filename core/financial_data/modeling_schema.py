from __future__ import annotations
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

class ValuationMethod(str, Enum):
    GORDON_GROWTH = "Gordon Growth"
    EXIT_MULTIPLE = "Exit Multiple"

class FinancialGlossary(BaseModel):
    """
    Definitions for key financial terms used in the modeling process.
    Acts as a source of truth for the agent's reasoning.
    """
    terms: Dict[str, str] = Field(default={
        "WACC": "Weighted Average Cost of Capital. The rate that a company is expected to pay on average to all its security holders to finance its assets.",
        "Free Cash Flow": "Operating Cash Flow minus Capital Expenditures. Represents the cash a company generates after accounting for cash outflows to support operations and maintain its capital assets.",
        "Terminal Value": "The value of an asset, business, or project beyond the forecasted period when future cash flows can be estimated.",
        "EBITDA": "Earnings Before Interest, Taxes, Depreciation, and Amortization. A proxy for operating cash flow.",
        "Sensitivity Analysis": "A technique used to determine how different values of an independent variable impact a particular dependent variable under a given set of assumptions."
    })

class FinancialAssumptions(BaseModel):
    """
    Input assumptions for financial modeling.
    """
    initial_cash_flow: float = Field(..., description="Starting Free Cash Flow")
    discount_rate: float = Field(..., description="WACC or required rate of return")
    growth_rate: float = Field(..., description="Projected annual growth rate for the forecast period")
    terminal_growth_rate: float = Field(..., description="Perpetual growth rate for terminal value")
    forecast_years: int = Field(10, description="Number of years to forecast")
    tax_rate: float = Field(0.21, description="Corporate tax rate")
    sentiment_adjustment_factor: float = Field(1.0, description="Multiplier derived from market sentiment (1.0 = neutral)")

class SensitivityScenario(BaseModel):
    name: str
    growth_rate_change: float = 0.0
    discount_rate_change: float = 0.0
    description: str

class ValuationResult(BaseModel):
    """
    The output of a valuation model.
    """
    intrinsic_value: float
    equity_value_per_share: Optional[float] = None
    implied_share_price: Optional[float] = None
    terminal_value: float
    present_value_of_cash_flows: float
    assumptions_used: FinancialAssumptions
    method: ValuationMethod
    sensitivity_results: Optional[Dict[str, float]] = None

class DiscountedCashFlowModel(BaseModel):
    """
    Structured representation of a DCF analysis.
    """
    company_id: str
    valuation_date: str
    assumptions: FinancialAssumptions
    projections: List[float] = Field(..., description="Projected Free Cash Flows")
    result: Optional[ValuationResult] = None
    glossary: FinancialGlossary = Field(default_factory=FinancialGlossary)

    model_config = ConfigDict(populate_by_name=True)
