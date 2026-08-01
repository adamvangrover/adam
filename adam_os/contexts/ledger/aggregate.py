from typing import List, Optional, Any
from pydantic import BaseModel, PrivateAttr, Field
import structlog
from adam_os.core.events import DomainEvent, LoanOriginated, AssetRevalued, DebtUpdated

logger = structlog.get_logger(__name__)

class FinancialEntity(BaseModel):
    """
    Aggregate root that rebuilds its state from events.
    Combines Pydantic validation with robust CQRS/Event Sourcing mechanics.
    """
    entity_id: str
    asset_value_usd: float = Field(default=0.0, ge=0)
    total_debt_usd: float = Field(default=0.0, ge=0)
    version: int = Field(default=0, ge=0)
    
    # Private attribute to track staged events without exposing them in REST/JSON schemas
    _uncommitted_events: List[DomainEvent] = PrivateAttr(default_factory=list)

    def __init__(self, **data: Any):
        super().__init__(**data)
        logger.info("initialized_financial_entity", entity_id=self.entity_id)

    @property
    def ltv(self) -> float:
        """Calculates the current Loan-to-Value ratio."""
        if self.asset_value_usd == 0.0:
            return 0.0
        return self.total_debt_usd / self.asset_value_usd

    def apply_event(self, event: DomainEvent, is_new: bool = True) -> None:
        """
        Applies a single domain event to mutate aggregate state.
        Open and expandable: ignores unhandled events without breaking.
        """
        if isinstance(event, LoanOriginated):
            # Defensive fetching allows for both branch's event schemas (e.g., principal_amount vs total_debt_usd)
            self.asset_value_usd = getattr(event, 'asset_value_usd', getattr(event, 'asset_value', 0.0))
            self.total_debt_usd = getattr(event, 'total_debt_usd', getattr(event, 'principal_amount', 0.0))
        
        elif isinstance(event, AssetRevalued):
            self.asset_value_usd = getattr(event, 'new_asset_value_usd', getattr(event, 'new_asset_value', self.asset_value_usd))
        
        elif isinstance(event, DebtUpdated):
            self.total_debt_usd = getattr(event, 'new_debt_amount', self.total_debt_usd)
        
        else:
            # We safely ignore unknown events to allow domain expansion without breaking legacy entities
            logger.debug("ignored_unhandled_event", event_type=type(event).__name__)
            pass

        self.version += 1

        if is_new:
            self._uncommitted_events.append(event)
            logger.debug("event_staged", event_type=type(event).__name__, entity_id=self.entity_id, version=self.version)

    @classmethod
    def load_from_history(cls, entity_id: str, events: List[DomainEvent]) -> 'FinancialEntity':
        """Rebuilds the aggregate from an event stream factory-style."""
        entity = cls(entity_id=entity_id)
        for event in events:
            entity.apply_event(event, is_new=False)
            
        logger.info("entity_rebuilt_from_history", entity_id=entity_id, final_version=entity.version)
        return entity

    def get_uncommitted_events(self) -> List[DomainEvent]:
        """Returns the list of uncommitted events ready to be persisted to the event store."""
        return self._uncommitted_events

    def clear_uncommitted_events(self) -> None:
        """Clears the list of uncommitted events after successful persistence."""
        self._uncommitted_events.clear()
        logger.debug("uncommitted_events_cleared", entity_id=self.entity_id)