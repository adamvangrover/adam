import os
import logging
from typing import Optional

# Configure basic logging if not already configured elsewhere
# This is a utility module, so it's good practice to ensure logging works
# even if the main application hasn't configured it yet, though typically
# logging is configured at the application entry point.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_api_key(key_name: str) -> Optional[str]:
    """
    Retrieves an API key from environment variables.

    Args:
        key_name (str): The name of the environment variable (e.g., "NEWS_API_KEY").

    Returns:
        Optional[str]: The API key if found, otherwise None.
    """
    api_key_value = os.environ.get(key_name)
    if api_key_value is None:
        logging.warning(f"API key '{key_name}' not found in environment variables.")
        return None

    # Optionally, add a check for empty string if that's considered invalid
    if not api_key_value.strip():
        logging.warning(f"API key '{key_name}' found in environment variables but is empty or whitespace.")
        return None

    return api_key_value


if __name__ == '__main__':
    # Example Usage and Test
    # To test this, you would need to set environment variables before running.
    # For example, in your terminal:
    # export TEST_API_KEY_EXISTS="your_actual_api_key_here"
    # export TEST_API_KEY_EMPTY="   "

    logging.info("Attempting to fetch 'TEST_API_KEY_EXISTS'...")
    key1 = get_api_key("TEST_API_KEY_EXISTS")
    if key1:
        logging.info(f"TEST_API_KEY_EXISTS: Found (value hidden for security)")
    else:
        logging.info("TEST_API_KEY_EXISTS: Not found or invalid.")

    logging.info("Attempting to fetch 'TEST_API_KEY_NONEXISTENT'...")
    key2 = get_api_key("TEST_API_KEY_NONEXISTENT")
    if key2:
        logging.info(f"TEST_API_KEY_NONEXISTENT: Found (value hidden for security)")
    else:
        logging.info("TEST_API_KEY_NONEXISTENT: Not found or invalid.")

    logging.info("Attempting to fetch 'TEST_API_KEY_EMPTY'...")
    key3 = get_api_key("TEST_API_KEY_EMPTY")
    if key3:
        logging.info(f"TEST_API_KEY_EMPTY: Found (value hidden for security)")
    else:
        logging.info("TEST_API_KEY_EMPTY: Not found or invalid.")

    # Example of how it might be used in other modules:
    # NEWS_API_KEY = get_api_key("NEWS_API_KEY")
    # if not NEWS_API_KEY:
    #     raise ValueError("NEWS_API_KEY is essential and not found.")
    # else:
    #     print("NEWS_API_KEY loaded successfully.")
