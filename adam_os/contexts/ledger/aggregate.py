from typing import List, Optional
import structlog
from adam_os.core.events import DomainEvent, LoanOriginated, AssetRevalued, DebtUpdated

logger = structlog.get_logger()

class FinancialEntity:
    def __init__(self, entity_id: str) -> None:
        self.entity_id = entity_id
        self.debt: float = 0.0
        self.asset_value: float = 0.0
        self.version: int = 0
        self._uncommitted_events: List[DomainEvent] = []
        logger.info("initialized_financial_entity", entity_id=entity_id)

    def load_from_history(self, events: List[DomainEvent]) -> None:
        """Rebuilds the state of the aggregate from a history of events."""
        for event in events:
            self.apply(event, is_new=False)

    def apply(self, event: DomainEvent, is_new: bool = True) -> None:
        """Applies an event to the aggregate and updates state."""
        if isinstance(event, LoanOriginated):
            self.debt = event.principal_amount
            self.asset_value = event.asset_value
        elif isinstance(event, AssetRevalued):
            self.asset_value = event.new_asset_value
        elif isinstance(event, DebtUpdated):
            self.debt = event.new_debt_amount
        else:
            # We ignore unknown events or events that don't affect core financial state directly for now.
            pass

        self.version += 1

        if is_new:
            self._uncommitted_events.append(event)

    def get_ltv(self) -> float:
        """Calculates the current Loan-to-Value ratio."""
        if self.asset_value == 0:
            return 0.0
        return self.debt / self.asset_value

    def get_uncommitted_events(self) -> List[DomainEvent]:
        """Returns the list of uncommitted events."""
        return self._uncommitted_events

    def clear_uncommitted_events(self) -> None:
        """Clears the list of uncommitted events after they have been saved to the event store."""
        self._uncommitted_events = []
