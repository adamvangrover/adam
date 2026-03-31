import os
import logging

logger = logging.getLogger(__name__)


def get_api_key(key_name: str) -> str | None:
    """
    Retrieves an API key from environment variables.

    Args:
        key_name (str): The name of the environment variable (e.g., "NEWS_API_KEY").

    Returns:
        str | None: The API key if found, otherwise None.
    """
    api_key_value = os.environ.get(key_name)
    if api_key_value is None:
        logger.warning(f"API key '{key_name}' not found in environment variables.")
        return None

    if not api_key_value.strip():
        logger.warning(f"API key '{key_name}' found in environment variables but is empty or whitespace.")
        return None

    return api_key_value
