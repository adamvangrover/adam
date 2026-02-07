# core/utils/token_utils.py

import os
import yaml
import tiktoken  # Use tiktoken for accurate token counting
import logging

# Configure logging (consider moving to a central location)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Counts the number of tokens in a string using tiktoken.

    Args:
        text: The string to count tokens in.
        encoding_name: The name of the encoding to use.  Defaults to "cl100k_base",
                       which is used by gpt-4, gpt-3.5-turbo, and text-embedding-ada-002.

    Returns:
        The number of tokens in the string.
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(text))
        return num_tokens
    except KeyError:
        logging.warning(f"Encoding '{encoding_name}' not found, using 'cl100k_base' as fallback.")
        encoding = tiktoken.get_encoding("cl100k_base")  # Default to cl100k_base
        num_tokens = len(encoding.encode(text))
        return num_tokens
    except Exception as e:
        logging.error(f"Error counting tokens: {e}")
        return 0  # Return 0 on error


def get_token_limit(config: dict) -> int:
    """
    Retrieves the token limit from the system configuration.

    Args:
        config: The system configuration dictionary.

    Returns:
        The token limit as an integer.  Defaults to 4096 if not found.
    """
    return config.get("token_limit", 4096)  # Default to 4096


def check_token_limit(text: str, config: dict, margin: int = 0) -> bool:
    """
    Checks if the number of tokens in a string is within the configured limit.

    Args:
        text: The string to check.
        config: The system configuration dictionary.
        margin: An optional margin to subtract from the token limit.

    Returns:
        True if the token count is within the limit (including the margin),
        False otherwise.
    """
    token_limit = get_token_limit(config)
    num_tokens = count_tokens(text)
    return num_tokens <= (token_limit - margin)


# Example Usage (and testing)
if __name__ == '__main__':
    from core.utils.config_utils import load_config  # Import load_config

    # Create a dummy config for testing
    dummy_config = {"token_limit": 100}  # Set a low limit for testing
    with open("test_config.yaml", "w") as f:
        yaml.dump(dummy_config, f)
    test_config = load_config("test_config.yaml")

    test_string = "This is a test string to count tokens."
    print(f"'{test_string}' has {count_tokens(test_string)} tokens.")

    if check_token_limit(test_string, test_config):
        print("Test string is within the token limit.")
    else:
        print("Test string exceeds the token limit.")

    long_string = "This is a very long string. " * 50  # Create a string that exceeds the limit
    if check_token_limit(long_string, test_config):
        print("Long string is within the token limit.")
    else:
        print("Long string exceeds the token limit.")

    if check_token_limit(long_string, test_config, margin=20):
        print("Long string is within the token limit (with margin).")
    else:
        print("Long string exceeds the token limit (with margin).")
    # Clean up
    os.remove("test_config.yaml")
