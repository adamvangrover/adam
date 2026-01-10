import logging
import json
import asyncio
from typing import Callable, Dict, Any
from core.system.message_bus.base import MessageBroker

logger = logging.getLogger(__name__)

# Try import pika or aio_pika
try:
    import aio_pika
    AIO_PIKA_AVAILABLE = True
except ImportError:
    AIO_PIKA_AVAILABLE = False

class RabbitMQBroker(MessageBroker):
    """
    RabbitMQ implementation of the Message Broker.
    """

    def __init__(self, url: str = "amqp://guest:guest@localhost/"):
        self.url = url
        self.connection = None
        self.channel = None

    async def connect(self):
        if not AIO_PIKA_AVAILABLE:
            logger.warning("aio_pika not installed. RabbitMQBroker running in mock mode.")
            return

        try:
            self.connection = await aio_pika.connect_robust(self.url)
            self.channel = await self.connection.channel()
            logger.info("Connected to RabbitMQ.")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")

    async def publish(self, topic: str, message: Dict[str, Any]):
        if not self.channel:
            logger.info(f"[Mock Publish] Topic: {topic}, Msg: {message}")
            return

        try:
            exchange = await self.channel.declare_exchange("adam_events", aio_pika.ExchangeType.TOPIC)
            body = json.dumps(message).encode()
            await exchange.publish(
                aio_pika.Message(body=body),
                routing_key=topic
            )
        except Exception as e:
            logger.error(f"Publish failed: {e}")

    async def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]):
        if not self.channel:
            logger.info(f"[Mock Subscribe] Registered handler for {topic}")
            return

        try:
            queue = await self.channel.declare_queue(topic, auto_delete=True)

            async def process_message(message: aio_pika.IncomingMessage):
                async with message.process():
                    body = json.loads(message.body.decode())
                    await handler(body)

            await queue.consume(process_message)
        except Exception as e:
            logger.error(f"Subscribe failed: {e}")

    async def close(self):
        if self.connection:
            await self.connection.close()
