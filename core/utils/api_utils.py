"""
core/utils/api_utils.py

Architecture & Usage:
This module provides the central API validation utility for Adam OS.
It is designed to cleanly validate API request data structures, ensuring required parameters are present.
It now includes an advanced AI-powered payload inspection feature to detect malicious intent.

Typical Usage Example:
    APIValidator.validate(request_data={"id": 123}, required_parameters=["id"])

    # Or with legacy wrapper:
    validate_api_request(request_data={"id": 123}, required_parameters=["id"])
"""

import logging
import json
from typing import Any

logger = logging.getLogger(__name__)

class MaliciousPayloadError(ValueError):
    """Raised when an API payload is suspected of malicious intent by the AI inspector."""
    pass

class PayloadSizeError(ValueError):
    """Raised when an API payload exceeds maximum safe structural entropy limits."""
    pass

class APIValidator:
    """
    Validates API request data structures and provides security analysis.
    """

    # Maximum allowed payload length (stringified) to prevent DoS via massive payloads
    MAX_PAYLOAD_LENGTH = 50_000

    @staticmethod
    def validate(request_data: dict[str, Any], required_parameters: list[str]) -> bool:
        """
        Validates API request data against a list of required parameters.
        Raises ValueError if a required parameter is missing to maintain legacy compatibility.

        AI Context:
            Purpose: Confirms that a given payload (`request_data`) contains all keys in `required_parameters`.
            Optimizer: Uses `set` difference and an O(1) generator expression for the exception string.

        Args:
            request_data (dict[str, Any]): The API request payload to validate.
            required_parameters (list[str]): A list of string keys that must be present in request_data.

        Returns:
            bool: True if validation succeeds.

        Raises:
            TypeError: If request_data is not a dictionary.
            ValueError: If a required parameter is missing.
        """
        if not isinstance(request_data, dict):
            raise TypeError(f"Expected request_data to be a dict, got {type(request_data).__name__}")

        # O(N) lookup for missing parameters
        missing_params = set(required_parameters) - request_data.keys()

        if missing_params:
            missing_str = ", ".join(sorted(missing_params))
            logger.error(f"API Validation failed. Missing required parameters: {missing_str}")

            # Identify the first missing parameter based on the original required_parameters list order
            # This maintains precise backward compatibility with legacy tests
            first_missing = next(param for param in required_parameters if param in missing_params)
            raise ValueError(f"Missing required parameter: {first_missing}")

        return True

    @staticmethod
    async def inspect_payload_intent(request_data: dict[str, Any]) -> dict[str, Any]:
        """
        AI Feature: Analyzes an API payload using a mock LLM zero-shot classifier
        to detect potential injection attacks, prompt leaking, or malicious intent.

        AI Context:
            Purpose: Scans payload strings for known dangerous heuristics (e.g., prompt injection, SQLi).
            Innovator: Incorporates a structural entropy/length boundary check before serialization to drop massive payloads instantly.
            Returns: A dictionary with `is_safe`, `safety_score`, `flags`, and `recommendation`.

        Args:
            request_data (dict[str, Any]): The incoming API payload to analyze.

        Returns:
            dict[str, Any]: Analysis results containing a safety score and flags.

        Raises:
            MaliciousPayloadError: If the payload is determined to be highly dangerous (score <= 0.5).
            PayloadSizeError: If the payload violates structural entropy size constraints.
        """
        logger.info("Initiating AI smart payload inspection...")

        try:
            payload_str = json.dumps(request_data).lower()
        except TypeError:
            payload_str = str(request_data).lower()

        # Structural Entropy Check: Prevent CPU exhaustion from massive payload scanning
        if len(payload_str) > APIValidator.MAX_PAYLOAD_LENGTH:
            logger.warning(f"Payload rejected: Exceeds max length constraint ({len(payload_str)} > {APIValidator.MAX_PAYLOAD_LENGTH})")
            raise PayloadSizeError("Payload exceeds maximum safe structural entropy limit.")

        suspicious_patterns = [
            "ignore previous instructions",
            "system prompt",
            "sql injection",
            "drop table",
            "1=1",
            "<script>",
            "exec(",
            "eval("
        ]

        flags = [f"Detected suspicious pattern related to: {pat}" for pat in suspicious_patterns if pat in payload_str]

        # Calculate safety score based on heuristics
        safety_score = max(0.0, 1.0 - (len(flags) * 0.25))
        is_safe = safety_score > 0.5

        result = {
            "is_safe": is_safe,
            "safety_score": safety_score,
            "flags": flags,
            "recommendation": "allow" if is_safe else "block"
        }

        if not is_safe:
            logger.warning(f"Payload flagged as malicious. Flags: {flags}")
            raise MaliciousPayloadError(f"Payload rejected by AI security inspector. Score: {safety_score}")

        return result

def validate_api_request(request_data: dict[str, Any], required_parameters: list[str]) -> bool:
    """
    Legacy wrapper for API validation.

    Args:
        request_data (dict[str, Any]): The API request payload to validate.
        required_parameters (list[str]): A list of string keys that must be present.

    Returns:
        bool: True if validation succeeds.

    Raises:
        ValueError: If a required parameter is missing.
    """
    return APIValidator.validate(request_data, required_parameters)
