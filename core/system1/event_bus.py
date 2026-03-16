import asyncio
import logging
from typing import Dict, Any, Callable, Awaitable

class EventBus:
    """
    A lightweight, asynchronous Pub/Sub Event Bus.
    Simulates a localized Redis instance to pass Pheromones (messages)
    between the rapid System 1 Swarm workers and the Pheromone Engine.
    """
    
    def __init__(self):
        # topic -> list of async queue objects
        self.subscribers: Dict[str, list[asyncio.Queue]] = {}
        logging.info("System 1 Event Bus Initialized.")

    def subscribe(self, topic: str) -> asyncio.Queue:
        """
        Creates a new queue subscribed to a specific topic.
        """
        if topic not in self.subscribers:
            self.subscribers[topic] = []
            
        queue = asyncio.Queue()
        self.subscribers[topic].append(queue)
        logging.info(f"New subscriber attached to topic: {topic}")
        return queue

    async def publish(self, topic: str, message: Dict[str, Any]):
        """
        Broadcasts a message to all queues subscribed to the topic.
        """
        if topic not in self.subscribers or not self.subscribers[topic]:
            # No listeners for this topic
            return
            
        # Push message strictly to async queues
        for queue in self.subscribers[topic]:
            await queue.put(message)
            
    def get_subscriber_count(self, topic: str) -> int:
        return len(self.subscribers.get(topic, []))
