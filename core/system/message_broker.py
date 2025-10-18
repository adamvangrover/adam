# core/system/message_broker.py

from abc import ABC, abstractmethod

class MessageBroker(ABC):
    """
    Abstract base class for message broker clients.
    """

    @abstractmethod
    def connect(self):
        """Connects to the message broker."""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnects from the message broker."""
        pass

    @abstractmethod
    def publish(self, topic: str, message: str):
        """Publishes a message to a specific topic."""
        pass

    @abstractmethod
    def subscribe(self, topic: str, callback):
        """Subscribes to a topic and registers a callback."""
        pass
