"""
Adam OS - Event Sourcing Domain Models

Defines deterministic payload schemas for specific financial events
and the aggregate state models they project into.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# --- Event Payload Models ---

class SecurityIssuancePayload(BaseModel):
    """Payload for when a new security is issued."""
    ticker: str = Field(..., description="The ticker symbol of the security.")
    asset_class: str = Field(..., description="e.g., 'Equity', 'Fixed Income', 'Derivative'")
    initial_shares: int = Field(0, description="Initial shares outstanding.")
    initial_price: float = Field(..., description="Issuance price.")

class RiskRatingPayload(BaseModel):
    """Payload for a change in risk rating."""
    agency: str = Field(..., description="Agency issuing the rating (e.g., 'Moody', 'S&P', 'Internal').")
    rating: str = Field(..., description="The alphanumeric rating (e.g., 'AAA', 'B-').")
    outlook: str = Field("Stable", description="Rating outlook (Positive, Stable, Negative).")

class TradingLevelPayload(BaseModel):
    """Payload for a change in trading level/volume/price."""
    price: float = Field(..., description="The current execution or quoted price.")
    volume: int = Field(0, description="Volume traded at this level.")
    liquidity_score: float = Field(1.0, description="Score 0.0 to 1.0 representing market depth.")

class PricingTargetPayload(BaseModel):
    """Payload for an analyst pricing target update."""
    analyst_id: str = Field(..., description="ID of the analyst or firm.")
    target_price: float = Field(..., description="The projected target price.")
    horizon_months: int = Field(12, description="Time horizon for the target in months.")

class MacroConditionPayload(BaseModel):
    """Payload for a shift in macroeconomic conditions."""
    indicator: str = Field(..., description="The macro indicator (e.g., 'Fed Funds Rate', 'CPI').")
    value: float = Field(..., description="The new value of the indicator.")
    regime: str = Field(..., description="The identified macro regime (e.g., 'Risk-On', 'Stagflation').")

class NewsTriggerPayload(BaseModel):
    """Payload for an external news event triggering system rules."""
    source: str = Field(..., description="News source.")
    headline: str = Field(..., description="News headline.")
    sentiment_score: float = Field(..., description="Sentiment from -1.0 to 1.0.")
    impact_severity: int = Field(1, description="Scale 1-5 of expected impact.")

# --- Aggregate State Models ---

class SecurityState(BaseModel):
    """
    The reconstructed state of a specific financial security.
    Built by replaying events against an empty state.
    """
    aggregate_id: str = Field(..., description="Usually the ticker or ISIN.")
    asset_class: str = Field("Unknown")
    shares_outstanding: int = Field(0)
    current_price: float = Field(0.0)
    risk_ratings: Dict[str, Dict[str, str]] = Field(default_factory=dict, description="Map of agency to rating details.")
    analyst_targets: Dict[str, float] = Field(default_factory=dict, description="Map of analyst to target price.")
    last_trading_volume: int = Field(0)
    news_sentiment_aggregate: float = Field(0.0)
    news_count: int = Field(0)

class MarketState(BaseModel):
    """
    The reconstructed state of the broader macroeconomic market.
    Built by replaying macro events.
    """
    aggregate_id: str = Field("global_market", description="The ID for the macro environment.")
    current_regime: str = Field("Unknown")
    indicators: Dict[str, float] = Field(default_factory=dict)
