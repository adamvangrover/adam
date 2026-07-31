from datetime import datetime, timezone
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class DomainEvent(BaseModel):
    """Base class for all domain events in the system."""
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    entity_id: str


class LoanOriginated(DomainEvent):
    """Event triggered when a new loan is originated."""
    principal_amount: float
    asset_value: float


class AssetRevalued(DomainEvent):
    """Event triggered when an underlying asset's valuation changes."""
    new_asset_value: float


class DebtUpdated(DomainEvent):
    """Event triggered when the debt amount changes (e.g. interest accrued, payment)."""
    new_debt_amount: float


class CovenantEvaluated(DomainEvent):
    """Event triggered when a financial covenant is evaluated."""
    covenant_type: str
    is_breached: bool
    evaluation_details: dict
