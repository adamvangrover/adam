import uuid
from typing import Any, Dict
from pydantic import BaseModel, Field

class DomainEvent(BaseModel):
    """Base class for all domain events."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class LoanOriginated(DomainEvent):
    """Event triggered when a new loan is originated."""
    event_type: str = "LoanOriginated"
    entity_id: str
    asset_value_usd: float
    total_debt_usd: float

class AssetRevalued(DomainEvent):
    """Event triggered when an asset is revalued."""
    event_type: str = "AssetRevalued"
    entity_id: str
    new_asset_value_usd: float

class CovenantEvaluated(DomainEvent):
    """Event triggered when a covenant is evaluated."""
    event_type: str = "CovenantEvaluated"
    entity_id: str
    covenant_name: str
    passed: bool
    evaluated_value: float
    threshold_value: float
    alert_triggered: bool

class CovenantBreachedEvent(DomainEvent):
    """Event triggered when a covenant is breached."""
    event_type: str = "CovenantBreachedEvent"
    entity_id: str
    covenant_name: str
    evaluated_value: float
    threshold_value: float
