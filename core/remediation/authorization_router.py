"""
Authoritative Model Authorization & Routing Quarantine Module (F-06, Gate G1).

Enforces:
1. Centralized model allowlist and denylist enforcement across all model aliases, fallbacks,
   direct instantiations, and tool execution boundaries.
2. Hard quarantine of blocked or unverified models.
3. Strict policy that LLM model consensus CANNOT grant fiduciary, capital, or execution authority.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelAuthorizationError(PermissionError):
    """Raised when an unauthorized, blocked, or quarantined model is requested or invoked."""
    pass


class FiduciaryAuthorityError(PermissionError):
    """Raised when model consensus or LLM output attempts to grant fiduciary execution authority."""
    pass


class ModelQuarantineState(str, Enum):
    ALLOWED = "ALLOWED"
    BLOCKED_UNTRUSTED = "BLOCKED_UNTRUSTED"
    QUARANTINED = "QUARANTINED"
    DEPRECATED = "DEPRECATED"


@dataclass
class ModelPolicyRegistry:
    allowed_models: set[str] = field(
        default_factory=lambda: {
            "gpt-4o",
            "claude-3-5-sonnet",
            "gemini-1-5-pro",
            "gpt-6-luna",
            "gemini-4-argon",
            "claude-sonnet-5-5",
        }
    )
    denied_models: set[str] = field(
        default_factory=lambda: {
            "deprecated-untrusted-v1",
            "unverified-experimental-v2",
            "blocked-legacy-model",
            "shadow-routing-alias",
        }
    )
    aliases: dict[str, str] = field(
        default_factory=lambda: {
            "default-fallback": "gpt-4o",
            "experimental-alias": "unverified-experimental-v2",
        }
    )


class AuthoritativeAuthorizationRouter:
    """
    Central routing and tool execution quarantine gatekeeper.
    """

    def __init__(self, policy: ModelPolicyRegistry | None = None) -> None:
        self.policy = policy or ModelPolicyRegistry()

    def resolve_model_identifier(self, model_identifier: str) -> str:
        """Resolves model aliases to their canonical model identifier."""
        canonical = self.policy.aliases.get(model_identifier, model_identifier)
        return canonical

    def check_model_authorization(self, model_identifier: str, execution_path: str = "direct") -> str:
        """
        Validates model authorization across every invocation boundary.
        """
        canonical = self.resolve_model_identifier(model_identifier)

        if canonical in self.policy.denied_models:
            raise ModelAuthorizationError(
                f"Model '{model_identifier}' (resolved: '{canonical}') is BLOCKED by quarantine policy "
                f"on execution path '{execution_path}'."
            )

        if canonical not in self.policy.allowed_models:
            raise ModelAuthorizationError(
                f"Model '{model_identifier}' (resolved: '{canonical}') is NOT on the authoritative allowlist "
                f"for execution path '{execution_path}'."
            )

        return canonical

    def execute_tool_call(
        self,
        model_identifier: str,
        tool_name: str,
        tool_args: dict[str, Any],
        fallback_models: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Executes a tool call after verifying authorization across primary and all fallback models.
        """
        # Validate primary model
        canonical_primary = self.check_model_authorization(model_identifier, execution_path=f"tool:{tool_name}:primary")

        # Validate all fallbacks beforehand
        if fallback_models:
            for fallback in fallback_models:
                self.check_model_authorization(fallback, execution_path=f"tool:{tool_name}:fallback")

        return {
            "status": "AUTHORIZED_EXECUTION",
            "resolved_model": canonical_primary,
            "tool_name": tool_name,
            "tool_args": tool_args,
        }

    def assert_fiduciary_boundary(self, consensus_score: float, action: str) -> None:
        """
        Enforces SR 11-7 invariant: Model consensus score CANNOT directly grant fiduciary or transaction authority.
        """
        if action in {"GRANT_CAPITAL", "APPROVE_LOAN", "EXECUTE_WIRE", "SANCTION_TRANSACTION"}:
            raise FiduciaryAuthorityError(
                f"Action '{action}' requires deterministic human-in-the-loop / governance authorization. "
                f"Model consensus score ({consensus_score}) cannot grant fiduciary authority."
            )
