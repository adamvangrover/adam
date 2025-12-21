from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field, validator


class DebtInstrument(BaseModel):
    name: str = Field(..., description="Official name of the facility, e.g., 'Term Loan B'")
    amount: float = Field(..., gt=0)
    interest_rate_spread: float = Field(..., description="Spread over SOFR in basis points")
    is_covenant_lite: bool = Field(default=False)

    @validator('interest_rate_spread')
    def check_spread(cls, v):
        if v > 2000: raise ValueError("Spread > 2000bps is likely an error")
        return v

class FinancialProfile(BaseModel):
    ticker: str
    debt_stack: List[DebtInstrument]
    ebitda_adjustments: List[Dict[str, Any]]

    @property
    def total_debt(self) -> float:
        return sum(d.amount for d in self.debt_stack)
