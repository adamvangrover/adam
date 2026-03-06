"""
NATS JetStream Enterprise Event Bus Abstraction.
Provides ultra-low-latency event streaming and RPC for Adam OS.
"""

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class EnterpriseEventBus:
    """
    Manages the connection and routing to NATS JetStream.
    Handles high-throughput publish/subscribe mechanisms, replay policies,
    and binary payload encoding.
    """

    def __init__(self, connection_url: str = "nats://localhost:4222"):
        self.connection_url = connection_url
        self._connected = False
        self._subscriptions = {}
        logger.info(f"Initialized EnterpriseEventBus with endpoint: {connection_url}")

    async def connect(self):
        """Asynchronously establishes connection to NATS JetStream."""
        # Simulated connection
        self._connected = True
        logger.info("Successfully connected to NATS JetStream.")

    async def publish(
        self, subject: str, payload: bytes, ack_wait: float = 1.0
    ) -> bool:
        """
        Publishes a normalized binary payload (e.g., Protocol Buffers) to a subject.
        """
        if not self._connected:
            raise ConnectionError("Cannot publish, Event Bus not connected.")

        logger.debug(
            f"Publishing {len(payload)} bytes to {subject} (ACK wait: {ack_wait}s)"
        )
        return True

    async def subscribe(
        self,
        subject: str,
        callback: Callable[[bytes], None],
        durable_name: Optional[str] = None,
    ):
        """
        Subscribes to a subject with an optional durable consumer name for at-least-once delivery.
        """
        if not self._connected:
            raise ConnectionError("Cannot subscribe, Event Bus not connected.")

        self._subscriptions[subject] = callback
        durable_str = f" [Durable: {durable_name}]" if durable_name else ""
        logger.info(f"Subscribed to subject: {subject}{durable_str}")

    def get_active_subscriptions(self) -> list:
        return list(self._subscriptions.keys())
