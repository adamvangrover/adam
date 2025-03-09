#core/utils/data_utils.py

import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
import pika
import csv
import logging
import yaml #if needed
from pathlib import Path
from typing import Dict, Any, Optional, Union, List


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# RabbitMQ connection parameters
RABBITMQ_HOST = 'localhost'  # Or your RabbitMQ server address
RABBITMQ_QUEUE = 'adam_data'  # A common queue for Adam data exchange


def clean_data(data, data_type):
    """
    Cleans data based on its type.

    Args:
      data: The data to be cleaned.
      data_type: The type of data (e.g., "text", "numerical", "time_series").

    Returns:
      The cleaned data.
    """
    if data_type == "text":
        return clean_text_data(data)
    elif data_type == "numerical":
        return clean_numerical_data(data)
    elif data_type == "time_series":
        return clean_time_series_data(data)
    else:
        raise ValueError("Invalid data type.")


def clean_text_data(text):
    """
    Cleans text data by removing irrelevant characters and formatting.

    Args:
      text: The text data to be cleaned.

    Returns:
      The cleaned text data.
    """
    # Remove special characters and punctuation
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = " ".join(text.split())
    return text


def clean_numerical_data(data):
    """
    Cleans numerical data by handling missing values and outliers.

    Args:
      data: The numerical data to be cleaned.

    Returns:
      The cleaned numerical data.
    """
    # Handle missing values (replace with mean, median, etc.)
    # Identify and handle outliers (e.g., remove, transform)
    #... (Implementation for cleaning numerical data)
    return data  # Placeholder for actual implementation


def clean_time_series_data(data):
    """
    Cleans time series data by handling missing values, smoothing, and detrending.

    Args:
      data: The time series data to be cleaned.

    Returns:
      The cleaned time series data.
    """
    # Handle missing values (interpolation, forward/backward fill, etc.)
    # Smooth the data (moving average, exponential smoothing, etc.)
    # Detrend the data (differencing, time series decomposition, etc.)
    #... (Implementation for cleaning time series data)
    return data  # Placeholder for actual implementation


def validate_data(data, data_type, constraints):
    """
    Validates data against specified constraints.

    Args:
      data: The data to be validated.
      data_type: The type of data (e.g., "text", "numerical", "time_series").
      constraints: A dictionary of constraints (e.g., {"min_value": 0, "max_value": 100}).

    Returns:
      True if the data is valid, False otherwise.
    """
    if data_type == "numerical":
        # Check if data is within the specified range
        if "min_value" in constraints and data < constraints["min_value"]:
            return False
        if "max_value" in constraints and data > constraints["max_value"]:
            return False
    elif data_type == "text":
        # Check if data matches a specific pattern (e.g., email format)
        if "pattern" in constraints and not re.match(constraints["pattern"], data):
            return False
    #... (Add more validation checks for other data types)
    return True


def transform_data(data, transformation_type):
    """
    Transforms data based on the specified transformation type.

    Args:
      data: The data to be transformed.
      transformation_type: The type of transformation (e.g., "standardize", "normalize", "log_transform").

    Returns:
      The transformed data.
    """
    if transformation_type == "standardize":
        # Standardize the data (mean=0, std=1)
        return (data - data.mean()) / data.std()
    elif transformation_type == "normalize":
        # Normalize the data (min=0, max=1)
        return (data - data.min()) / (data.max() - data.min())
    elif transformation_type == "log_transform":
        # Apply log transformation to the data
        return np.log(data)
    else:
        raise ValueError("Invalid transformation type.")


def convert_to_datetime(date_str, format="%Y-%m-%d"):
    """
    Converts a date string to a datetime object.

    Args:
      date_str: The date string to be converted.
      format: The format of the date string (default: "%Y-%m-%d").

    Returns:
      A datetime object representing the date.
    """
    return datetime.strptime(date_str, format)


def convert_to_dataframe(data, columns=None):
    """
    Converts data to a Pandas DataFrame.

    Args:
      data: The data to be converted (e.g., list of lists, dictionary).
      columns: A list of column names (optional).

    Returns:
      A Pandas DataFrame.
    """
    return pd.DataFrame(data, columns=columns)


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

# Add more data utility functions as needed

def load_data(source_config: Dict[str, Any]) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Loads data from a file based on the provided configuration.

    Args:
        source_config: A dictionary containing the 'type' (json, yaml, csv) and 'path' of the data source.

    Returns:
        The loaded data, or None if an error occurred.
    """
    try:
        file_type = source_config.get("type")
        file_path = source_config.get("path")

        if not file_type or not file_path:
            logging.error("Invalid source configuration: 'type' and 'path' are required.")
            return None

        if file_type == "json":
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_type == "yaml":
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        elif file_type == "csv":
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                return list(reader)
        else:
            logging.error(f"Unsupported file type: {file_type}")
            return None
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.exception(f"Error loading data from {file_path}: {e}")
        return None



    except (json.JSONDecodeError, IOError, yaml.YAMLError) as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None
# Example Usage: You would call it from within an agent, like this:
#   data_sources = load_config("config/data_sources.yaml")
#   risk_data = load_data(data_sources['risk_ratings'])



