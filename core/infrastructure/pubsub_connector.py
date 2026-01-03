from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable
import logging
import json
import asyncio

logger = logging.getLogger(__name__)

class PubSubConnector:
    """
    Standardized interface for Google Cloud Pub/Sub interactions.
    Facilitates asynchronous, event-driven architecture for the Alphabet Ecosystem.
    """

    def __init__(self, project_id: str, topic_id: str, subscription_id: str = None):
        self.project_id = project_id
        self.topic_id = topic_id
        self.subscription_id = subscription_id
        self.publisher = None
        self.subscriber = None
        self._init_clients()

    def _init_clients(self):
        try:
            # from google.cloud import pubsub_v1
            # self.publisher = pubsub_v1.PublisherClient()
            # if self.subscription_id:
            #     self.subscriber = pubsub_v1.SubscriberClient()
            logger.info(f"PubSubConnector initialized for {self.topic_id} (Stub)")
        except ImportError:
            logger.warning("google-cloud-pubsub not installed. Using mock behavior.")

    async def publish(self, message: Dict[str, Any], attributes: Dict[str, str] = None) -> str:
        """
        Publishes a message to the topic.
        """
        data_str = json.dumps(message)
        data = data_str.encode("utf-8")
        attributes = attributes or {}

        logger.info(f"[PubSub Stub] Publishing to {self.topic_id}: {data_str[:50]}... Attrs: {attributes}")

        # Simulate network delay
        await asyncio.sleep(0.1)

        return "msg_id_stub_pub_123"

    def listen(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Starts listening for messages (Blocking/Callback-based).
        """
        if not self.subscription_id:
            raise ValueError("Subscription ID required for listening.")

        logger.info(f"[PubSub Stub] Listening on {self.subscription_id}...")
        # In a real app, this would start the streaming pull future.
