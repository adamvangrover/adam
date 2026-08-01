import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict

class DomainEvent(BaseModel):
    """
    Base class for all domain events in the system. 
    Designed for event-sourced ledgers with built-in accessibility and expandability.
    """
    # Allows dynamic fields to be attached and enables alias matching from both branches
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = Field(..., description="Explicit discriminant for safe deserialization")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    entity_id: str
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Open payload for custom operational data without altering the schema"
    )

class LoanOriginated(DomainEvent):
    """Event triggered when a new loan is originated."""
    event_type: str = Field(default="LoanOriginated", frozen=True)
    
    # Aliases map HEAD's naming into main's strict USD naming standard
    asset_value_usd: float = Field(..., alias="asset_value")
    total_debt_usd: float = Field(..., alias="principal_amount")

class AssetRevalued(DomainEvent):
    """Event triggered when an underlying asset's valuation changes."""
    event_type: str = Field(default="AssetRevalued", frozen=True)
    new_asset_value_usd: float = Field(..., alias="new_asset_value")

class DebtUpdated(DomainEvent):
    """Event triggered when the debt amount changes (e.g., interest accrued, payment)."""
    # From HEAD branch - added for complete aggregate support
    event_type: str = Field(default="DebtUpdated", frozen=True)
    new_debt_amount: float

class CovenantEvaluated(DomainEvent):
    """
    Event triggered when a financial covenant is evaluated.
    Merges detailed audit payloads (HEAD) with strict metrics (main).
    """
    event_type: str = Field(default="CovenantEvaluated", frozen=True)
    covenant_name: str = Field(..., alias="covenant_type")
    
    # Dual boolean flags support different downstream consumer logic
    passed: bool
    is_breached: bool
    alert_triggered: bool
    
    # Optional strict metrics
    evaluated_value: Optional[float] = None
    threshold_value: Optional[float] = None
    
    # Deep forensic payload
    evaluation_details: Dict[str, Any] = Field(default_factory=dict)

class CovenantBreachedEvent(DomainEvent):
    """Event triggered specifically when a covenant is actively breached."""
    # From main branch - used for strict metric alerts
    event_type: str = Field(default="CovenantBreachedEvent", frozen=True)
    covenant_name: str
    evaluated_value: float
    threshold_value: float