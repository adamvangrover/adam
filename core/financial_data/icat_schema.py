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

class LBOParameters(BaseModel):
    entry_multiple: float = Field(..., description="EV/EBITDA entry multiple")
    exit_multiple: float = Field(..., description="EV/EBITDA exit multiple")
    equity_contribution_percent: float = Field(..., description="Equity contribution as % of Total Sources")
    debt_structure: List[DebtTranche] = Field(..., description="List of debt tranches")
    transaction_fees: float = Field(0.0, description="Transaction fees")
    tax_rate: float = Field(0.21, description="Corporate tax rate")
    capex_percent_revenue: float = Field(0.05, description="Capex as % of Revenue")
    working_capital_percent_revenue: float = Field(0.10, description="Working Capital as % of Revenue")
    forecast_years: int = Field(5, description="Number of years for projection")

class LBOResult(BaseModel):
    irr: float = Field(..., description="Internal Rate of Return")
    mom_multiple: float = Field(..., description="Money-on-Money Multiple (MOIC)")
    equity_value_entry: float = Field(..., description="Equity value at entry")
    equity_value_exit: float = Field(..., description="Equity value at exit")
    debt_paydown: float = Field(..., description="Total debt paid down over hold period")

class CreditMetrics(BaseModel):
    pd_1yr: float = Field(..., description="1-Year Probability of Default")
    lgd: float = Field(..., description="Loss Given Default")
    ltv: float = Field(..., description="Loan to Value Ratio")
    dscr: float = Field(..., description="Debt Service Coverage Ratio")
    interest_coverage: float = Field(..., description="EBIT / Interest Expense")
    net_leverage: float = Field(..., description="Net Debt / EBITDA")
    z_score: Optional[float] = Field(None, description="Altman Z-Score")

class ValuationMetrics(BaseModel):
    enterprise_value: float = Field(..., description="Enterprise Value")
    equity_value: float = Field(..., description="Equity Value")
    dcf_value: Optional[float] = Field(None, description="Value derived from DCF")
    trading_comps_value: Optional[float] = Field(None, description="Value derived from Trading Comps")
    transaction_comps_value: Optional[float] = Field(None, description="Value derived from Transaction Comps")

class CarveOutParameters(BaseModel):
    parent_entity: str = Field(..., description="Parent Company Name")
    spin_off_segment: str = Field(..., description="Segment/Division to be carved out")
    standalone_cost_adjustments: float = Field(0.0, description="Estimated standalone costs (negative impact on EBITDA)")
    tax_leakage: float = Field(0.0, description="Tax leakage estimation")

class ICATOutput(BaseModel):
    ticker: str
    scenario_name: str
    credit_metrics: CreditMetrics
    valuation_metrics: ValuationMetrics
    lbo_analysis: Optional[LBOResult] = None
    carve_out_impact: Optional[float] = None # Impact on valuation
    generated_at: str

    model_config = ConfigDict(populate_by_name=True)
