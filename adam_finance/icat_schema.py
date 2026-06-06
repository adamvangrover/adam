from __future__ import annotations
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

class DebtTranche(BaseModel):
    name: str = Field(..., description="Name of the debt tranche (e.g., Senior Term Loan A)")
    amount: float = Field(..., description="Principal amount")
    interest_rate: float = Field(..., description="Interest rate (e.g., 0.05 for 5%)")
    amortization_rate: float = Field(0.0, description="Annual amortization as % of principal")
    maturity_years: int = Field(5, description="Years to maturity")
    cash_sweep_share: float = Field(0.0, description="Percentage of excess cash flow swept to pay down this tranche (0.0 to 1.0)")
    pik_interest: float = Field(0.0, description="Payment-in-Kind interest component (added to principal)")
    seniority: int = Field(1, description="Seniority level (1 = Senior Secured, higher is more junior)")

    def model_copy(self):
        return self.copy()

class EnvironmentContext(BaseModel):
    risk_free_rate: float = Field(0.04, description="Risk-free rate (e.g., 10Y Treasury)")
    market_risk_premium: float = Field(0.06, description="Market Risk Premium")
    inflation_rate: float = Field(0.02, description="Expected inflation rate")

class ForecastAssumptions(BaseModel):
    revenue_growth: List[float] = Field(..., description="List of annual revenue growth rates")
    ebitda_margin: List[float] = Field(..., description="List of annual EBITDA margins")
    capex_percent_revenue: float = Field(0.05, description="Capex as % of Revenue")
    working_capital_percent_revenue: float = Field(0.10, description="Working Capital as % of Revenue")
    tax_rate: float = Field(0.21, description="Corporate tax rate")
    terminal_growth_rate: float = Field(0.02, description="Perpetual growth rate for terminal value")
    discount_rate: Optional[float] = Field(None, description="WACC (if None, calculated from EnvironmentContext)")

class LBOParameters(BaseModel):
    entry_multiple: float = Field(..., description="EV/EBITDA entry multiple")
    exit_multiple: float = Field(..., description="EV/EBITDA exit multiple")
    equity_contribution_percent: float = Field(..., description="Equity contribution as % of Total Sources")
    debt_structure: List[DebtTranche] = Field(..., description="List of debt tranches")
    transaction_fees: float = Field(0.0, description="Transaction fees")
    forecast_years: int = Field(5, description="Number of years for projection")

class LBOResult(BaseModel):
    irr: float = Field(..., description="Internal Rate of Return")
    mom_multiple: float = Field(..., description="Money-on-Money Multiple (MOIC)")
    equity_value_entry: float = Field(..., description="Equity value at entry")
    equity_value_exit: float = Field(..., description="Equity value at exit")
    debt_paydown: float = Field(..., description="Total debt paid down over hold period")
    exit_enterprise_value: float = Field(..., description="Enterprise Value at exit")
    exit_equity_value: float = Field(..., description="Equity Value at exit")

class CreditMetrics(BaseModel):
    pd_1yr: float = Field(..., description="1-Year Probability of Default")
    lgd: float = Field(..., description="Loss Given Default")
    ltv: float = Field(..., description="Loan to Value Ratio")
    dscr: float = Field(..., description="Debt Service Coverage Ratio (minimum)")
    avg_dscr: float = Field(..., description="Average DSCR over forecast period")
    interest_coverage: float = Field(..., description="EBIT / Interest Expense")
    net_leverage: float = Field(..., description="Net Debt / EBITDA")
    z_score: Optional[float] = Field(None, description="Altman Z-Score")
    credit_rating: Optional[str] = Field(None, description="Implied Credit Rating")

class ValuationMetrics(BaseModel):
    enterprise_value: float = Field(..., description="Enterprise Value (Current)")
    equity_value: float = Field(..., description="Equity Value (Current)")
    dcf_value: Optional[float] = Field(None, description="Value derived from DCF")
    trading_comps_value: Optional[float] = Field(None, description="Value derived from Trading Comps")
    transaction_comps_value: Optional[float] = Field(None, description="Value derived from Transaction Comps")
    terminal_value_method: str = Field("Gordon Growth", description="Method used for Terminal Value")

class CarveOutParameters(BaseModel):
    parent_entity: str = Field(..., description="Parent Company Name")
    spin_off_segment: str = Field(..., description="Segment/Division to be carved out")
    standalone_cost_adjustments: float = Field(0.0, description="Estimated standalone costs (negative impact on EBITDA)")
    tax_leakage: float = Field(0.0, description="Tax leakage estimation")
    tsa_cost: float = Field(0.0, description="Transition Service Agreement cost per year")
    tsa_duration: int = Field(0, description="Duration of TSA in years")
    stranded_costs: float = Field(0.0, description="Costs remaining at parent (informational)")

class ICATOutput(BaseModel):
    ticker: str
    scenario_name: str
    environment: EnvironmentContext
    credit_metrics: CreditMetrics
    valuation_metrics: ValuationMetrics
    lbo_analysis: Optional[LBOResult] = None
    carve_out_impact: Optional[float] = None # Impact on valuation
    generated_at: str

    model_config = ConfigDict(populate_by_name=True)
