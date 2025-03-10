# core/utils/config_utils.py

import yaml
import os  # Import the 'os' module
import logging
from typing import Dict, Any, Optional
from core.system.error_handler import ConfigurationError


# Configure logging (consider moving to a central location)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# core/utils/config_utils.py
import yaml
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads a YAML configuration file.

    Args:
        file_path: The path to the YAML configuration file.

    Returns:
        The configuration as a dictionary, or None if an error occurred.
    """
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
            if config is None: # Handle empty YAML files
                logging.warning(f"Configuration file is empty: {config_path}")
                return {} # Return empty dict
            return config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML in {config_path}: {e}")
        return {}  # Return an empty dictionary on error
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading config: {e}")
        return {}
    except FileNotFoundError:
        logging.error(f"Config file not found: {file_path}")
        return None
    except Exception as e:
        logging.exception(f"Error loading config from {file_path}: {e}")
        return None
        
# Example Usage (you can remove this when you integrate it into your project)
if __name__ == '__main__':
    # Create a dummy config file for testing
    dummy_config = {"key1": "value1", "key2": 123}
    with open("test_config.yaml", "w") as f:
        yaml.dump(dummy_config, f)

    # Load the dummy config
    config = load_config("test_config.yaml")
    print(f"Loaded config: {config}")

    # Test with a non-existent file
    config = load_config("nonexistent_file.yaml")
    print(f"Loaded config (nonexistent): {config}")

    # Test with an invalid YAML file.
    with open("invalid.yaml", "w") as f:
        f.write("key1: value1\nkey2: - item1\n  -item2")  # Invalid YAML
    config = load_config("invalid.yaml")
    print(f"Loaded config (invalid YAML): {config}")

    # Test with an empty yaml file
    with open("empty.yaml", "w") as f:
        pass # Creates an empty file
    config = load_config("empty.yaml")
    print(f"Loaded config (empty YAML): {config}")

    # Cleanup the test files
    os.remove("test_config.yaml")
    os.remove("invalid.yaml")
    os.remove("empty.yaml")
