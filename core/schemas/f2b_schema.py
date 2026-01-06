from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Literal, Union
from datetime import datetime
from uuid import UUID, uuid4

class FinancialInstrument(BaseModel):
    """
    Canonical definition of a Financial Instrument.
    """
    id: str = Field(..., description="ISIN, CUSIP, or internal ID")
    type: Literal["equity", "bond", "swap", "option", "future", "etf", "fx"]
    symbol: str
    currency: str = "USD"
    notional_value: float = Field(..., description="Face value or notional amount")
    maturity_date: Optional[datetime] = None
    issuer: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)

class Trade(BaseModel):
    """
    Canonical definition of a Trade event.
    """
    id: UUID = Field(default_factory=uuid4)
    instrument: FinancialInstrument
    quantity: float
    price: float
    timestamp: datetime = Field(default_factory=datetime.now)
    counterparty_id: str
    direction: Literal["buy", "sell"]
    status: Literal["pending", "executed", "settled", "failed"]

    model_config = ConfigDict(populate_by_name=True)

class Counterparty(BaseModel):
    """
    Canonical definition of a Counterparty.
    """
    id: str
    name: str
    credit_limit: float
    current_exposure: float
    risk_rating: str
    status: Literal["active", "suspended", "defaulted", "probation"]
    sector: Optional[str] = None
    jurisdiction: Optional[str] = None

class PortfolioState(BaseModel):
    """
    Real-time snapshot of a portfolio state.
    """
    id: str
    timestamp: datetime
    holdings: Dict[str, float] = Field(..., description="Map of Instrument ID to Quantity")
    total_value: float
    risk_metrics: Dict[str, float] = Field(default_factory=dict, description="VaR, CVaR, PFE, etc.")
    metadata: Dict[str, str] = Field(default_factory=dict)

class BreachEvent(BaseModel):
    """
    Structure for a Credit Limit Breach event triggering RCA.
    """
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime
    counterparty_id: str
    breach_amount: float
    limit: float
    exposure_at_breach: float

    # Context data for Branch A, B, C analysis
    recent_trades: List[Trade] = Field(default_factory=list)
    market_volatility_index: float = 0.0
    collateral_status: Literal["good", "pending", "failed"] = "good"

class RCAOutput(BaseModel):
    """
    Structured output from the Root Cause Analysis LLM.
    """
    event_id: str
    primary_cause: Literal["new_trade", "market_movement", "collateral_failure", "systemic_error"]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    narrative: str
    recommended_action: Literal["soft_block", "hard_block", "increase_limit", "margin_call"]
    branch_scores: Dict[str, float] = Field(..., description="Scores for Branch A, B, C")
