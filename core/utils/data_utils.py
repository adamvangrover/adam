# core/utils/data_utils.py

def clean_data(data):
    # Placeholder for data cleaning logic
    print("Cleaning data...")
    # Simulated cleaned data
    cleaned_data = data  # For now, just return the original data
    return cleaned_data

def format_data(data, format_type):
    # Placeholder for data formatting logic
    print(f"Formatting data to {format_type}...")
    # Simulated formatted data
    formatted_data = data  # For now, just return the original data
    return formatted_data

# core/utils/data_utils.py

import pika
import json

# RabbitMQ connection parameters
RABBITMQ_HOST = 'localhost'  # Or your RabbitMQ server address
RABBITMQ_QUEUE = 'adam_data'  # A common queue for Adam data exchange

def send_message(message, queue=RABBITMQ_QUEUE):
    """
    Sends a message to a RabbitMQ queue.

    Args:
        message (dict): The message to send (will be serialized to JSON).
        queue (str, optional): The name of the queue. Defaults to RABBITMQ_QUEUE.
    """
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
        channel = connection.channel()
        channel.queue_declare(queue=queue)
        channel.basic_publish(exchange='', routing_key=queue, body=json.dumps(message))
        connection.close()
        print(f"Sent message to queue '{queue}': {message}")
    except Exception as e:
        print(f"Error sending message to RabbitMQ: {e}")

def receive_messages(queue=RABBITMQ_QUEUE, callback=None):
    """
    Receives messages from a RabbitMQ queue.

    Args:
        queue (str, optional): The name of the queue. Defaults to RABBITMQ_QUEUE.
        callback (function, optional): A function to call when a message is received.
                                        Defaults to None (prints the message).
    """
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
        channel = connection.channel()
        channel.queue_declare(queue=queue)

        if callback is None:
            def default_callback(ch, method, properties, body):
                print(f"Received message from queue '{queue}': {body}")
            callback = default_callback

        channel.basic_consume(queue=queue, on_message_callback=callback, auto_ack=True)
        print(f"Waiting for messages from queue '{queue}'...")
        channel.start_consuming()
    except Exception as e:
        print(f"Error receiving messages from RabbitMQ: {e}")
