# core/system/message_broker.py

import logging
from collections import defaultdict
from threading import RLock
from typing import Callable, Dict, List, Optional

class MessageBroker:
    _instance: Optional['MessageBroker'] = None
    _lock = RLock()

    def __init__(self):
        self.topics: Dict[str, List[Callable]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    @classmethod
    def get_instance(cls) -> 'MessageBroker':
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = cls()
        return cls._instance

    def subscribe(self, topic: str, callback: Callable):
        with self._lock:
            self.topics[topic].append(callback)
            self.logger.info(f"Subscribed to topic '{topic}'")

    def publish(self, topic: str, message: str):
        with self._lock:
            if topic in self.topics:
                for callback in self.topics[topic]:
                    try:
                        callback(message)
                    except Exception as e:
                        self.logger.error(f"Error in callback for topic '{topic}': {e}")
            self.logger.info(f"Published message to topic '{topic}'")

    def connect(self):
        self.logger.info("In-memory message broker connected.")

    def disconnect(self):
        self.logger.info("In-memory message broker disconnected.")
