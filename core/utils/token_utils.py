# core/utils/token_utils.py

import logging
from functools import lru_cache
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)


class TokenManager:
    """
    Manages the encoding, counting, and truncation of text tokens for LLM interactions.
    Encapsulates tiktoken interactions with resilient caching and robust fallback mechanisms.
    """

    DEFAULT_ENCODING = "cl100k_base"
    DEFAULT_TOKEN_LIMIT = 4096

    @classmethod
    @lru_cache(maxsize=32)
    def get_encoding(cls, encoding_name: str) -> tiktoken.Encoding:
        """
        Retrieves a tiktoken encoding with LRU caching.

        Args:
            encoding_name: The name of the encoding standard to retrieve (e.g., 'cl100k_base').

        Returns:
            A tiktoken.Encoding instance. Defaults to 'cl100k_base' if the specified encoding fails.
        """
        try:
            return tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(
                f"Failed to get encoding '{encoding_name}': {e}. Falling back to '{cls.DEFAULT_ENCODING}'."
            )
            return tiktoken.get_encoding(cls.DEFAULT_ENCODING)

    @classmethod
    @lru_cache(maxsize=128)
    def get_encoding_for_model(cls, model_name: str) -> tiktoken.Encoding:
        """
        Retrieves the specific tiktoken encoding for a given LLM model with LRU caching.

        Args:
            model_name: The name of the target model (e.g., 'gpt-4o', 'gpt-3.5-turbo').

        Returns:
            A tiktoken.Encoding instance optimized for the model, falling back to default if unrecognized.
        """
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(
                f"Model '{model_name}' not found for encoding. Falling back to '{cls.DEFAULT_ENCODING}'."
            )
            return tiktoken.get_encoding(cls.DEFAULT_ENCODING)
        except Exception as e:
            logger.error(
                f"Unexpected error retrieving encoding for model '{model_name}': {e}. Falling back."
            )
            return tiktoken.get_encoding(cls.DEFAULT_ENCODING)

    @classmethod
    def count_tokens(
        cls,
        text: str,
        encoding_name: str = DEFAULT_ENCODING,
        model_name: str | None = None,
    ) -> int:
        """
        Accurately calculates the exact token count of a given text string.

        Args:
            text: The raw text string to be tokenized.
            encoding_name: The fallback encoding standard to use.
            model_name: If provided, dynamically resolves the precise encoding required for that model.

        Returns:
            The precise number of tokens consumed by the text. Returns 0 on failure.
        """
        if not text or not isinstance(text, str):
            return 0

        try:
            encoding = (
                cls.get_encoding_for_model(model_name)
                if model_name
                else cls.get_encoding(encoding_name)
            )
            return len(encoding.encode(text))
        except Exception as e:
            logger.error(f"Critical error counting tokens: {e}")
            return 0

    @classmethod
    def get_token_limit(cls, config: dict[str, Any] | None) -> int:
        """
        Extracts the maximum permitted token context limit from a system configuration.

        Args:
            config: A configuration dictionary.

        Returns:
            The integer token limit, defaulting to 4096.
        """
        if not config or not isinstance(config, dict):
            return cls.DEFAULT_TOKEN_LIMIT
        return config.get("token_limit", cls.DEFAULT_TOKEN_LIMIT)

    @classmethod
    def check_token_limit(
        cls, text: str, config: dict[str, Any] | None, margin: int = 0
    ) -> bool:
        """
        Validates whether a given text string fits within the system's configured token limits.

        Args:
            text: The payload text to validate.
            config: The system configuration defining 'token_limit'.
            margin: Buffer of tokens to reserve (e.g., for the model's output generation).

        Returns:
            True if the payload is within the permitted limits.
        """
        limit = cls.get_token_limit(config)
        return cls.count_tokens(text) <= (limit - margin)

    @classmethod
    def truncate_text(
        cls,
        text: str,
        max_tokens: int,
        encoding_name: str = DEFAULT_ENCODING,
        model_name: str | None = None,
    ) -> str:
        """
        Intelligently truncates text so it fits exactly within a specified token limit.

        Args:
            text: The text to potentially truncate.
            max_tokens: The absolute maximum number of tokens allowed.
            encoding_name: The fallback encoding.
            model_name: The specific model context.

        Returns:
            The truncated string. If text is already under the limit, it is returned unchanged.
        """
        if not text or not isinstance(text, str):
            return ""

        if max_tokens <= 0:
            return ""

        try:
            encoding = (
                cls.get_encoding_for_model(model_name)
                if model_name
                else cls.get_encoding(encoding_name)
            )
            tokens = encoding.encode(text)

            if len(tokens) <= max_tokens:
                return text

            return encoding.decode(tokens[:max_tokens])
        except Exception as e:
            logger.error(f"Failed to safely truncate text: {e}")
            return text[: max_tokens * 4]  # Extremely rough fallback


# --- Backward Compatible Global Wrappers ---


def _get_encoding(encoding_name: str) -> tiktoken.Encoding:
    """Legacy wrapper for TokenManager.get_encoding."""
    return TokenManager.get_encoding(encoding_name)


def _get_encoding_for_model(model_name: str) -> tiktoken.Encoding:
    """Legacy wrapper for TokenManager.get_encoding_for_model."""
    return TokenManager.get_encoding_for_model(model_name)


def count_tokens(
    text: str,
    encoding_name: str = TokenManager.DEFAULT_ENCODING,
    model_name: str | None = None,
) -> int:
    """Legacy wrapper for TokenManager.count_tokens."""
    return TokenManager.count_tokens(text, encoding_name, model_name)


def get_token_limit(config: dict[str, Any] | None) -> int:
    """Legacy wrapper for TokenManager.get_token_limit."""
    return TokenManager.get_token_limit(config)


def check_token_limit(
    text: str, config: dict[str, Any] | None, margin: int = 0
) -> bool:
    """Legacy wrapper for TokenManager.check_token_limit."""
    return TokenManager.check_token_limit(text, config, margin)


def truncate_text(
    text: str,
    max_tokens: int,
    encoding_name: str = TokenManager.DEFAULT_ENCODING,
    model_name: str | None = None,
) -> str:
    """Legacy wrapper for TokenManager.truncate_text."""
    return TokenManager.truncate_text(text, max_tokens, encoding_name, model_name)
