# core/utils/token_utils.py

import logging
from functools import lru_cache
from typing import Optional, Dict, Any

import tiktoken

# Configure logging for the module
logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _get_encoding(encoding_name: str) -> tiktoken.Encoding:
    """
    Cached retrieval of a tiktoken encoding.

    Args:
        encoding_name: The name of the encoding to retrieve (e.g., 'cl100k_base').

    Returns:
        A tiktoken.Encoding object.

    Raises:
        KeyError: If the encoding name is completely unrecognized by tiktoken.
    """
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Failed to get encoding '{encoding_name}': {e}. Falling back to 'cl100k_base'.")
        return tiktoken.get_encoding("cl100k_base")

@lru_cache(maxsize=128)
def _get_encoding_for_model(model_name: str) -> tiktoken.Encoding:
    """
    Cached retrieval of a tiktoken encoding by model name.

    Args:
        model_name: The name of the model to retrieve (e.g., 'gpt-4o', 'gpt-3.5-turbo').

    Returns:
        A tiktoken.Encoding object.
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(f"Model '{model_name}' not found for encoding, falling back to 'cl100k_base'.")
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoding_name: str = "cl100k_base", model_name: Optional[str] = None) -> int:
    """
    Accurately calculates the token count of a given text string, essential for
    managing AI context windows and preventing LLM API limit errors.

    If a `model_name` is provided (e.g., 'gpt-4'), it dynamically resolves the precise
    encoding required for that specific model. Otherwise, it defaults to the robust
    'cl100k_base' encoding or a specified `encoding_name`.

    Args:
        text (str): The raw text string to be tokenized.
        encoding_name (str): The fallback encoding standard to use. Defaults to 'cl100k_base'.
        model_name (str, optional): The target LLM model (e.g., 'gpt-4o'). If set, this overrides `encoding_name`.

    Returns:
        int: The precise number of tokens the text consumes. Returns 0 if an unrecoverable error occurs.
    """
    if not text:
        return 0

    try:
        if model_name:
            encoding = _get_encoding_for_model(model_name)
        else:
            encoding = _get_encoding(encoding_name)

        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Critical error counting tokens: {e}")
        return 0


def get_token_limit(config: Dict[str, Any]) -> int:
    """
    Extracts the maximum permitted token context limit from a system configuration dictionary.

    Args:
        config (Dict[str, Any]): The configuration dictionary, typically loaded from YAML.

    Returns:
        int: The integer token limit, defaulting to 4096 if unspecified in the configuration.
    """
    return config.get("token_limit", 4096)


def check_token_limit(text: str, config: Dict[str, Any], margin: int = 0) -> bool:
    """
    Validates whether a given text string fits within the system's configured token limits,
    accounting for an optional safety margin.

    This is critical for pre-flight checks before sending massive payloads to an LLM API.

    Args:
        text (str): The payload text to validate.
        config (Dict[str, Any]): The system configuration dictionary defining 'token_limit'.
        margin (int): A buffer of tokens to subtract from the maximum limit (e.g., reserving space for the model's output).

    Returns:
        bool: True if the text length (in tokens) is less than or equal to the allowed limit minus the margin.
    """
    token_limit = get_token_limit(config)
    num_tokens = count_tokens(text)
    return num_tokens <= (token_limit - margin)
