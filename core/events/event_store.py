from typing import List
from core.events.envelope import EventEnvelope

class EventStore:
    def __init__(self):
        self._events: List[EventEnvelope] = []
        self._correlation_index = {}  # ⚡ Bolt: O(1) lookup index for correlation IDs

    def append(self, event: EventEnvelope):
        self._events.append(event)
        if event.correlation_id not in self._correlation_index:
            self._correlation_index[event.correlation_id] = []
        self._correlation_index[event.correlation_id].append(event)

    def get_stream(self) -> List[EventEnvelope]:
        return self._events.copy()

    def get_by_correlation_id(self, correlation_id: str) -> List[EventEnvelope]:
        return list(self._correlation_index.get(correlation_id, []))
