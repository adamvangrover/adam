from typing import List, Optional
from pydantic import BaseModel
from adam_os.core.events import DomainEvent, LoanOriginated, AssetRevalued

class FinancialEntity(BaseModel):
    """Aggregate root that rebuilds its state from events."""
    entity_id: str
    asset_value_usd: float = 0.0
    total_debt_usd: float = 0.0

    @property
    def ltv(self) -> float:
        if self.asset_value_usd == 0.0:
            return 0.0
        return self.total_debt_usd / self.asset_value_usd

    def apply_event(self, event: DomainEvent):
        """Applies a single domain event to mutate aggregate state."""
        if isinstance(event, LoanOriginated):
            if self.asset_value_usd == 0.0 and self.total_debt_usd == 0.0:
                self.asset_value_usd = event.asset_value_usd
                self.total_debt_usd = event.total_debt_usd
        elif isinstance(event, AssetRevalued):
            self.asset_value_usd = event.new_asset_value_usd

    @classmethod
    def load_from_history(cls, entity_id: str, events: List[DomainEvent]) -> 'FinancialEntity':
        """Rebuilds the aggregate from an event stream."""
        entity = cls(entity_id=entity_id)
        for event in events:
            entity.apply_event(event)
        return entity
