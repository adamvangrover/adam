import asyncio
from typing import Callable, Awaitable, Dict, List
from core.events.envelope import EventEnvelope

class NatsEventBusMock:
    """
    Mock implementation of NATS JetStream event bus for local testing and architectural validation.
    """
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[EventEnvelope], Awaitable[None]]]] = {}
        self.published_events: List[EventEnvelope] = []

    async def publish(self, subject: str, event: EventEnvelope):
        self.published_events.append(event)
        if subject in self.subscribers:
            for handler in self.subscribers[subject]:
                await handler(event)

    def subscribe(self, subject: str, handler: Callable[[EventEnvelope], Awaitable[None]]):
        if subject not in self.subscribers:
            self.subscribers[subject] = []
        self.subscribers[subject].append(handler)
