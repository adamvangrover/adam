import abc
import asyncio
import logging
from typing import Callable, Any, Dict

logger = logging.getLogger(__name__)

class MessageBus(abc.ABC):
    @abc.abstractmethod
    async def publish(self, topic: str, message: Dict[str, Any]):
        pass

    @abc.abstractmethod
    async def subscribe(self, topic: str, callback: Callable):
        pass

class InMemoryMessageBus(MessageBus):
    """
    Simple In-Memory implementation for testing/local dev.
    """
    def __init__(self):
        self.subscribers = {}

    async def publish(self, topic: str, message: Dict[str, Any]):
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                # Fire and forget (or await, depending on semantics)
                asyncio.create_task(callback(message))

    async def subscribe(self, topic: str, callback: Callable):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

# In production, we would implement RabbitMQMessageBus or RedisMessageBus here.
