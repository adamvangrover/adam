"""
core/utils/token_utils.py

Architecture & Usage:
This module provides a suite of utilities for accurate token counting and dynamic context window management.
It utilizes the `tiktoken` library for precise token calculation compatible with OpenAI models,
augmented with an LRU cache (`_get_encoding`) to eliminate redundant encoding initializations
and improve execution speed.

It provides functions to verify text against configuration-defined token limits (`check_token_limit`),
and introduces a novel AI integration (`trim_to_token_limit_with_ai`) that leverages `litellm`
to intelligently summarize verbose text blocks when limits are exceeded, falling back to safe
hard truncation if the LLM is unavailable.

Typical Usage Example:
    from core.utils.token_utils import count_tokens, trim_to_token_limit_with_ai

    token_count = count_tokens("Hello world")
    config = {"token_limit": 100}
    safe_text = trim_to_token_limit_with_ai(large_text, config)

Dependencies:
    - Built-ins: functools, logging, typing
    - External: tiktoken, litellm (optional)
"""

import logging
from functools import lru_cache
from typing import Any, Dict

import tiktoken  # Use tiktoken for accurate token counting

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _get_encoding(encoding_name: str) -> tiktoken.Encoding:
    """
    Cached helper to retrieve tiktoken encoding.
    """
    try:
        return tiktoken.get_encoding(encoding_name)
    except (KeyError, ValueError):
        logger.warning(f"Encoding '{encoding_name}' not found, using 'cl100k_base' as fallback.")
        return tiktoken.get_encoding("cl100k_base")


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
        encoding = _get_encoding(encoding_name)
        num_tokens = len(encoding.encode(text))
        return num_tokens
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return 0  # Return 0 on error


def get_token_limit(config: Dict[str, Any]) -> int:
    """
    Retrieves the token limit from the system configuration.

    Args:
        config: The system configuration dictionary.

    Returns:
        The token limit as an integer.  Defaults to 4096 if not found.
    """
    return config.get("token_limit", 4096)  # Default to 4096


def check_token_limit(text: str, config: Dict[str, Any], margin: int = 0) -> bool:
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


def trim_to_token_limit_with_ai(text: str, config: Dict[str, Any], margin: int = 0) -> str:
    """
    Intelligently summarizes or truncates text using litellm if it exceeds the token limit.
    If the text is within the limit, it is returned unmodified.

    Args:
        text: The string to check and potentially summarize.
        config: The system configuration dictionary.
        margin: An optional margin to subtract from the token limit.

    Returns:
        The text, summarized if it exceeded the token limit.
    """
    import os
    try:
        import litellm
    except ImportError:
        logger.warning("litellm not installed, falling back to truncation")
        litellm = None

    token_limit = get_token_limit(config)
    target_limit = token_limit - margin
    num_tokens = count_tokens(text)

    if num_tokens <= target_limit:
        return text

    if litellm is not None and os.environ.get("OPENAI_API_KEY"):
        try:
            logger.info(f"Text exceeds limit ({num_tokens} > {target_limit}), attempting AI summarization.")
            response = litellm.completion(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"Summarize the following text so that it is less than "
                                                  f"{target_limit} tokens while retaining key information."},
                    {"role": "user", "content": text}
                ],
                max_tokens=target_limit
            )
            summarized_text = response.choices[0].message.content
            if summarized_text:
                return summarized_text
        except Exception as e:
            logger.error(f"AI summarization failed: {e}. Falling back to hard truncation.")

    # Fallback: Graceful compaction using tiktoken
    logger.info(f"Falling back to graceful compaction to {target_limit} tokens.")
    import re

    # Step 1: Whitespace Compaction
    compacted_text = re.sub(r'\s+', ' ', text).strip()
    num_tokens = count_tokens(compacted_text)

    if num_tokens <= target_limit:
        return compacted_text

    # Step 2: Middle-out Truncation
    encoding = _get_encoding("cl100k_base")
    encoded = encoding.encode(compacted_text)

    half_limit = target_limit // 2
    if half_limit > 3:
        # Keep the beginning and end, inserting an ellipsis token sequence if there's room
        ellipsis = encoding.encode(" ... ")
        half_limit = (target_limit - len(ellipsis)) // 2
        truncated_encoded = encoded[:half_limit] + ellipsis + encoded[-half_limit:]
    else:
        # If target limit is too small, just hard truncate the beginning
        truncated_encoded = encoded[:target_limit]

    return encoding.decode(truncated_encoded)
