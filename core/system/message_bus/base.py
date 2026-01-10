from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

class MessageBroker(ABC):
    """
    Abstract Base Class for the Asynchronous Message Bus.
    """

    @abstractmethod
    async def publish(self, topic: str, message: Dict[str, Any]):
        """Publish a message to a topic."""
        pass

    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]):
        """Subscribe to a topic with a handler function."""
        pass

    @abstractmethod
    async def connect(self):
        """Establish connection to the broker."""
        pass

    @abstractmethod
    async def close(self):
        """Close the connection."""
        pass
