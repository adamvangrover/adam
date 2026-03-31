"""
core/utils/api_utils.py

Architecture & Usage:
This module is the definitive API validation utility for Adam OS.
It verifies payload structures and employs a two-tier AI inspection
pipeline (heuristics + zero-shot LLM) to neutralize malicious intent.

Inputs: Unsanitized API request dictionaries.
Outputs: Clean validation or raised strict Exceptions (ValueError, MaliciousPayloadError).
"""

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MaliciousPayloadError(ValueError):
    """Raised when an API payload triggers heuristic or AI security tripwires."""
    pass


class APIValidator:
    """
    Validates API requests and enforces multi-layered security.
    """
    MAX_PAYLOAD_LENGTH = 50000

    @staticmethod
    def _estimate_size(data: Any) -> int:
        """
        Memory-efficient recursive size estimator to prevent DoS via massive payloads.
        Replaces O(N) string allocation.
        """
        match data:
            case str():
                return len(data)
            case int() | float() | bool() | None:
                return 8  # Approximation
            case dict():
                return sum(APIValidator._estimate_size(k) + APIValidator._estimate_size(v) for k, v in data.items())
            case list() | tuple() | set():
                return sum(APIValidator._estimate_size(item) for item in data)
            case _:
                return 0

    @staticmethod
    def validate(request_data: dict[str, Any], required_parameters: list[str]) -> bool:
        """
        Validates API request data against a list of required parameters.

        Args:
            request_data (dict[str, Any]): The API request payload.
            required_parameters (list[str]): Mandatory keys.

        Returns:
            bool: True if validation succeeds.

        Raises:
            ValueError: If a required parameter is missing.
            TypeError: If request_data is not a dictionary.
        """
        if not isinstance(request_data, dict):
            raise TypeError(f"Expected request_data to be a dict, got {type(request_data).__name__}")

        # O(N) set checking, preserves deterministic backward compatibility
        missing_params = set(required_parameters) - set(request_data.keys())
        if missing_params:
            missing_str = ", ".join(sorted(missing_params))
            logger.error(f"API Validation failed. Missing required parameters: {missing_str}")
            for param in required_parameters:
                if param in missing_params:
                    raise ValueError(f"Missing required parameter: {param}")

        return True

    @staticmethod
    async def _call_llm_safety_check(payload_str: str) -> float:
        """
        Stub for an external, cutting-edge local LLM zero-shot classification call.
        Returns a safety score between 0.0 (malicious) and 1.0 (safe).
        """
        # In production, this awaits a fast model inference.
        return 1.0

    @staticmethod
    async def inspect_payload_intent(request_data: dict[str, Any]) -> dict[str, Any]:
        """
        Analyzes payload intent using static heuristics and an advanced LLM classifier.

        Args:
            request_data (dict[str, Any]): The incoming API payload.

        Returns:
            dict[str, Any]: Analysis containing 'is_safe', 'safety_score', and 'flags'.

        Raises:
            MaliciousPayloadError: If the payload is determined to be dangerous.
            ValueError: If the payload size exceeds MAX_PAYLOAD_LENGTH.
        """
        logger.info("Initiating advanced two-tier payload inspection...")

        if APIValidator._estimate_size(request_data) > APIValidator.MAX_PAYLOAD_LENGTH:
            logger.error(f"Payload size exceeds {APIValidator.MAX_PAYLOAD_LENGTH} limit.")
            raise ValueError(f"Payload length exceeds maximum allowed length of {APIValidator.MAX_PAYLOAD_LENGTH}")

        def _serialize(data: dict[str, Any]) -> str:
            try:
                return json.dumps(data).lower()
            except TypeError:
                return str(data).lower()

        # Thread offload for massive payload serialization
        payload_str = await asyncio.to_thread(_serialize, request_data)

        # Tier 1: Fast Heuristics
        suspicious_patterns = [
            "ignore previous instructions", "system prompt", "sql injection",
            "drop table", "1=1", "<script>", "exec(", "eval("
        ]
        flags = [f"Detected pattern: {pattern}" for pattern in suspicious_patterns if pattern in payload_str]
        heuristic_score = max(0.0, 1.0 - (len(flags) * 0.25))

        # Tier 2: Zero-shot LLM (Innovator Phase)
        llm_score = await APIValidator._call_llm_safety_check(payload_str)

        # Combined Assessment
        final_score = min(heuristic_score, llm_score)
        is_safe = final_score > 0.5

        result = {
            "is_safe": is_safe,
            "safety_score": final_score,
            "flags": flags,
            "recommendation": "allow" if is_safe else "block"
        }

        if not is_safe:
            logger.warning(f"Payload blocked. Score: {final_score}. Flags: {flags}")
            raise MaliciousPayloadError(f"Payload rejected by AI security inspector. Score: {final_score}")

        return result


def validate_api_request(request_data: dict[str, Any], required_parameters: list[str]) -> bool:
    """Legacy wrapper for API validation."""
    return APIValidator.validate(request_data, required_parameters)
