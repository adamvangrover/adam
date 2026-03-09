# core/utils/data_utils.py

import json
import csv
import logging
import yaml
from pathlib import Path
try:
    import pika
except ImportError:
    pika = None
from typing import Dict, Any, Optional, Union, List
from core.system.error_handler import FileReadError, InvalidInputError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# RabbitMQ connection parameters
RABBITMQ_HOST = 'localhost'
RABBITMQ_QUEUE = 'adam_data'

# Simple in-memory cache
_data_cache: Dict[str, Any] = {}

def send_message(message, queue=RABBITMQ_QUEUE):
    """
    Sends a message to a RabbitMQ queue.
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

class DataLoader:
    """
    A class to handle loading data from files (JSON, CSV, YAML) or APIs,
    with built-in caching.
    """
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.supported_types = ["json", "csv", "yaml"]

    def load(self, source_config: Dict[str, Any]) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """Loads data based on the provided configuration."""
        data_type = source_config.get("type")
        file_path = source_config.get("path")

        if data_type == "api":
            return self._load_api(source_config)

        if not file_path:
            logging.error("Invalid source configuration: 'path' is required for file-based sources.")
            raise InvalidInputError("Missing 'path' in source_config")

        if data_type not in self.supported_types:
            logging.error(f"Unsupported data type: {data_type}")
            raise InvalidInputError(f"Unsupported data type: {data_type}")

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            logging.error(f"Data file not found: {file_path_obj}")
            raise FileReadError(str(file_path_obj), "File not found")

        cache_key = str(file_path_obj)
        if self.use_cache and cache_key in _data_cache:
            logging.info(f"Loading data from cache: {file_path_obj}")
            return _data_cache[cache_key]

        data = None
        try:
            if data_type == "json":
                data = self._load_json(file_path_obj)
            elif data_type == "csv":
                data = self._load_csv(file_path_obj)
            elif data_type == "yaml":
                data = self._load_yaml(file_path_obj)

            if self.use_cache and data is not None:
                _data_cache[cache_key] = data
            return data

        except (json.JSONDecodeError, FileNotFoundError, IOError, yaml.YAMLError) as e:
            logging.exception(f"Error loading data from {file_path_obj}: {e}")
            raise FileReadError(str(file_path_obj), str(e)) from e
        except Exception as e:
            logging.exception(f"Unexpected error loading data from {file_path_obj}: {e}")
            raise FileReadError(str(file_path_obj), str(e)) from e

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        with file_path.open('r') as f:
            return json.load(f)

    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        with file_path.open('r', newline='') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        with file_path.open('r') as f:
            return yaml.safe_load(f)

    def _load_api(self, source_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Provides placeholder data for API calls."""
        logging.warning("API data source not yet implemented. Returning placeholder data.")
        provider = source_config.get("provider")

        if provider == "example_financial_data_api":
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
            return {
                "market_trends": [
                   {"sector": "healthcare", "trend": "neutral"},
                   {"sector": "energy", "trend": "downward"}
                ]
            }
        else:
            logging.warning(f"Unknown API provider: {provider}")
            return None


def load_data(source_config: Dict[str, Any], cache: bool = True) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Wrapper for DataLoader to maintain backward compatibility.
    Loads data from a file (JSON or CSV) or a placeholder for an API, based on configuration.
    """
    loader = DataLoader(use_cache=cache)
    return loader.load(source_config)
