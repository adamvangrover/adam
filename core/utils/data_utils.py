#core/utils/data_utils.py

import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
try:
    import pika
except ImportError:
    pika = None
import csv
import logging
import yaml #if needed
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from core.system.error_handler import FileReadError, InvalidInputError, DataNotFoundError


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# RabbitMQ connection parameters
RABBITMQ_HOST = 'localhost'  # Or your RabbitMQ server address
RABBITMQ_QUEUE = 'adam_data'  # A common queue for Adam data exchange

# Simple in-memory cache (for demonstration - use a proper library like 'cachetools' for production)
_data_cache: Dict[str, Any] = {}


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
    if pika is None:
        print(f"[Mock] Sent message to queue '{queue}': {message}")
        return

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
    if pika is None:
        print(f"[Mock] Waiting for messages from queue '{queue}'... (Pika not installed)")
        return

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


def load_data(source_config: Dict[str, Any], cache: bool = True) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Loads data from a file (JSON or CSV) or a placeholder for an API, based on configuration.

    Args:
        source_config: A dictionary with 'type', 'path' (for files), 'provider' (for APIs).
        cache: Whether to use the in-memory cache.

    Returns:
        The loaded data, or None if an error occurs.
    """
    data_type = source_config.get("type")
    file_path = source_config.get("path")
    provider = source_config.get("provider")  # For API calls

    if data_type == "api":
        # Placeholder for API calls.  In a real implementation, you'd have separate
        # API client classes (as discussed before) to handle the specifics of each API.
        logging.warning("API data source not yet implemented. Returning placeholder data.")
        return _get_api_placeholder_data(source_config)  # Use a helper function

    if not file_path:
        logging.error("Invalid source configuration: 'path' is required for file-based sources.")
        raise InvalidInputError("Missing 'path' in source_config")

    file_path = Path(file_path)
    if not file_path.exists():
        logging.error(f"Data file not found: {file_path}")
        raise FileReadError(str(file_path), "File not found")

    cache_key = str(file_path)  # Use string representation of Path as cache key
    if cache and cache_key in _data_cache:
        logging.info(f"Loading data from cache: {file_path}")
        return _data_cache[cache_key]

    try:
        if data_type == "json":
            with file_path.open('r') as f:
                data = json.load(f)
                if cache:
                    _data_cache[cache_key] = data
                return data
        elif data_type == "csv":
            with file_path.open('r', newline='') as f:
                reader = csv.DictReader(f)
                data = list(reader)
                if cache:
                    _data_cache[cache_key] = data
                return data
        elif data_type == 'yaml': #Adding yaml
            with file_path.open('r') as f:
                data = yaml.safe_load(f)
                if cache:
                    _data_cache[cache_key] = data
                return data
        else:
            logging.error(f"Unsupported data type: {data_type}")
            raise InvalidInputError(f"Unsupported data type: {data_type}")

    except (json.JSONDecodeError, FileNotFoundError, IOError, yaml.YAMLError) as e:
        logging.exception(f"Error loading data from {file_path}: {e}")
        raise FileReadError(str(file_path), str(e)) from e
    except Exception as e: #catch any unexpected errors
        logging.exception(f"Unexpected error loading data from {file_path}: {e}")
        raise FileReadError(str(file_path), str(e)) from e

def _get_api_placeholder_data(source_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Provides placeholder data for API calls (replace with actual API calls).

    This function simulates the responses you might get from different API providers.
    """
    provider = source_config.get("provider")
    #TODO: Include a comprehensive list of all possible data requests, and appropriate formatting to ensure correct responses.
    if provider == "example_financial_data_api":  # Replace with a real provider name
        # Simulate fetching financial statements
        return {
            "income_statement": {
                "revenue": [1000, 1100, 1250],
                "net_income": [100, 120, 150],
            },
            "balance_sheet": {
                "assets": [2000, 2100, 2200],
                "liabilities": [800, 850, 900],
            },
            "cash_flow_statement": {
                "operating_cash_flow": [150, 170, 200]
            }
        }
    elif provider == "example_market_data_api":
          return{
             "market_trends":[
                {"sector": "healthcare", "trend": "neutral"},
                {"sector": "energy", "trend": "downward"}
              ]
          }

    # Add more providers and data as needed
    else:
        logging.warning(f"Unknown API provider: {provider}")
        return None

