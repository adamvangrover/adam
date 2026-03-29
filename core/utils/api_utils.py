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

Dependencies:
    - Built-ins: asyncio, json, logging, typing
"""

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MaliciousPayloadError(ValueError):
    """Raised when an API payload is suspected of malicious intent by the AI inspector."""
    pass


class APIValidator:
    """
    Validates API request data structures and provides security analysis.
    """

    @staticmethod
    def validate(request_data: dict[str, Any], required_parameters: list[str]) -> bool:
        """
        Validates API request data against a list of required parameters.
        Raises ValueError if a required parameter is missing to maintain legacy compatibility.

        Args:
            request_data (dict[str, Any]): The API request payload to validate.
            required_parameters (list[str]): A list of string keys that must be present in request_data.

        Returns:
            bool: True if validation succeeds.

        Raises:
            ValueError: If a required parameter is missing.
            TypeError: If request_data is not a dictionary.
        """
        if not isinstance(request_data, dict):
            raise TypeError(f"Expected request_data to be a dict, got {type(request_data).__name__}")

        # Use set difference for O(N) checking instead of O(N*M) loop
        missing_params = set(required_parameters) - set(request_data.keys())
        if missing_params:
            # Sort to ensure deterministic error messages
            missing_str = ", ".join(sorted(missing_params))
            logger.error(f"API Validation failed. Missing required parameters: {missing_str}")
            # Maintain exact legacy exception format for backward compatibility
            # Legacy expected "Missing required parameter: param" for the first missing param
            # We'll just raise for the first one we find in the required_parameters list to match exactly
            for param in required_parameters:
                if param in missing_params:
                    raise ValueError(f"Missing required parameter: {param}")

        return True

    @staticmethod
    async def inspect_payload_intent(request_data: dict[str, Any]) -> dict[str, Any]:
        """
        AI Feature: Analyzes an API payload using a mock LLM zero-shot classifier
        to detect potential injection attacks, prompt leaking, or malicious intent.

        In production, this would route to a local safety model or fast LLM endpoint.
        Uses `asyncio.to_thread` for JSON serialization to avoid blocking the event loop on massive payloads.

        Args:
            request_data (dict[str, Any]): The incoming API payload to analyze.

        Returns:
            dict[str, Any]: Analysis results containing a safety score and flags.

        Raises:
            MaliciousPayloadError: If the payload is determined to be highly dangerous.
        """
        logger.info("Initiating AI smart payload inspection...")

        def _serialize_payload(data: dict[str, Any]) -> str:
            """Helper to serialize payload off the main thread."""
            try:
                return json.dumps(data).lower()
            except TypeError:
                return str(data).lower()

        # Optimize: offload CPU-heavy JSON serialization to a background thread
        payload_str = await asyncio.to_thread(_serialize_payload, request_data)

        # Mock LLM Safety Heuristics
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

        flags = [f"Detected suspicious pattern related to: {pattern}" for pattern in suspicious_patterns if pattern in payload_str]

        # Safety calculation
        safety_score = max(0.0, 1.0 - (len(flags) * 0.25))

        result = {
            "is_safe": safety_score > 0.5,
            "safety_score": safety_score,
            "flags": flags,
            "recommendation": "allow" if safety_score > 0.5 else "block"
        }

        if not result["is_safe"]:
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
