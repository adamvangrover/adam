from typing import List
from core.events.envelope import EventEnvelope

class EventStore:
    def __init__(self):
        self._events: List[EventEnvelope] = []

    def append(self, event: EventEnvelope):
        self._events.append(event)

    def get_stream(self) -> List[EventEnvelope]:
        return self._events.copy()

    def get_by_correlation_id(self, correlation_id: str) -> List[EventEnvelope]:
        return [e for e in self._events if e.correlation_id == correlation_id]
