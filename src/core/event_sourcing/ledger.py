"""
Adam OS - Event Sourcing Core (CQRS)

Provides an immutable, append-only ledger for all financial transactions and
system state transitions, ensuring replayability and strict auditability.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field
import structlog
from src.schemas.core_types import compute_deterministic_hash

logger = structlog.get_logger(__name__)

class FinancialEvent(BaseModel):
    """Immutable event representing a single state transition in the system."""
    event_id: str = Field(..., description="Unique deterministic identifier for the event")
    aggregate_id: str = Field(..., description="The ID of the entity this event applies to (e.g., 'portfolio_1')")
    event_type: str = Field(..., description="The type of the event (e.g., 'ASSET_PURCHASED')")
    payload: Dict[str, Any] = Field(default_factory=dict, description="The deterministic data of the event")
    timestamp: str = Field(..., description="Deterministic timestamp of the event")
    event_hash: str = Field("", description="Deterministic hash of the event payload and metadata")

    def model_post_init(self, __context: Any) -> None:
        """Computes the hash of the event upon creation to ensure immutability."""
        if not self.event_hash:
            data_to_hash = {
                "aggregate_id": self.aggregate_id,
                "event_type": self.event_type,
                "payload": self.payload,
                "timestamp": self.timestamp
            }
            self.event_hash = compute_deterministic_hash(data_to_hash)

class EventLedger:
    """
    In-memory, append-only event ledger for storing and replaying events.
    In a full production environment, this would be backed by a durable store like Kafka or EventStoreDB.
    """
    def __init__(self):
        self._events: List[FinancialEvent] = []

    def append_event(self, event: FinancialEvent) -> None:
        """Appends a new immutable event to the ledger."""
        # Simple validation: ensure the hash matches the payload to detect tampering before append
        data_to_hash = {
            "aggregate_id": event.aggregate_id,
            "event_type": event.event_type,
            "payload": event.payload,
            "timestamp": event.timestamp
        }
        expected_hash = compute_deterministic_hash(data_to_hash)
        if event.event_hash != expected_hash:
            logger.error("event_tampering_detected", event_id=event.event_id, expected=expected_hash, actual=event.event_hash)
            raise ValueError(f"Event {event.event_id} failed integrity check. Hash mismatch.")

        self._events.append(event)
        logger.info("event_appended", event_id=event.event_id, aggregate_id=event.aggregate_id, event_type=event.event_type)

    def get_events_for_aggregate(self, aggregate_id: str) -> List[FinancialEvent]:
        """Retrieves all events for a specific aggregate, ordered by insertion."""
        return [e for e in self._events if e.aggregate_id == aggregate_id]

    def replay_aggregate(self, aggregate_id: str, reducer: callable, initial_state: Any) -> Any:
        """
        Reconstructs the state of an aggregate by replaying its events through a reducer function.

        :param aggregate_id: The entity to reconstruct.
        :param reducer: A function that takes (current_state, event) and returns the new state.
        :param initial_state: The starting state before any events are applied.
        """
        events = self.get_events_for_aggregate(aggregate_id)
        state = initial_state
        for event in events:
            state = reducer(state, event)
        logger.info("aggregate_replayed", aggregate_id=aggregate_id, event_count=len(events))
        return state

    def clear(self) -> None:
        """Clears the ledger. Strictly for testing purposes."""
        self._events = []
