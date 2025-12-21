# core/system/brokers/rabbitmq_client.py

import pika

from core.system.message_broker import MessageBroker


class RabbitMQClient(MessageBroker):
    """
    An implementation of the MessageBroker for RabbitMQ.
    """

    def __init__(self, host='localhost'):
        self.host = host
        self.connection = None
        self.channel = None

    def connect(self):
        """Connects to the RabbitMQ server."""
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))
        self.channel = self.connection.channel()

    def disconnect(self):
        """Disconnects from the RabbitMQ server."""
        if self.connection:
            self.connection.close()

    def publish(self, topic: str, message: str):
        """Publishes a message to a specific topic."""
        self.channel.queue_declare(queue=topic)
        self.channel.basic_publish(exchange='',
                                   routing_key=topic,
                                   body=message)

    def subscribe(self, topic: str, callback):
        """Subscribes to a topic and registers a callback."""
        self.channel.queue_declare(queue=topic)
        self.channel.basic_consume(queue=topic,
                                   on_message_callback=callback,
                                   auto_ack=True)
        self.channel.start_consuming()
