"""
Policy hooks and execution guardrails for Antigravity SDK.
"""

import fnmatch
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class PolicyAction:
    ALLOW = "ALLOW"
    DENY = "DENY"
    ASK_USER = "ASK_USER"
    ENFORCE = "ENFORCE"


@dataclass
class PolicyRule:
    pattern: str
    action: str
    handler: Optional[Callable[..., Any]] = None
    enforce_kwargs: Optional[Dict[str, Any]] = None


def allow(pattern: str) -> PolicyRule:
    """Policy rule allowing tools matching pattern."""
    return PolicyRule(pattern=pattern, action=PolicyAction.ALLOW)


def deny(pattern: str) -> PolicyRule:
    """Policy rule blocking tools matching pattern."""
    return PolicyRule(pattern=pattern, action=PolicyAction.DENY)


def ask_user(pattern: str, handler: Optional[Callable[..., Any]] = None) -> PolicyRule:
    """Policy rule requiring user confirmation before tool invocation."""
    return PolicyRule(pattern=pattern, action=PolicyAction.ASK_USER, handler=handler)


def enforce(pattern: str, **kwargs: Any) -> PolicyRule:
    """Policy rule enforcing static arguments for tool invocation."""
    return PolicyRule(pattern=pattern, action=PolicyAction.ENFORCE, enforce_kwargs=kwargs)


class HookRunner:
    """
    Evaluates policy rules against tool call attempts before execution.
    """

    def __init__(self, policies: Optional[List[PolicyRule]] = None) -> None:
        self.policies: List[PolicyRule] = policies or []

    def evaluate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> PolicyRule:
        """
        Evaluates policies in reverse order (last rule takes precedence).
        Defaults to ALLOW if no policies match.
        """
        for policy in reversed(self.policies):
            if fnmatch.fnmatch(tool_name, policy.pattern):
                logger.info("Matched policy %s for tool %s", policy.action, tool_name)
                return policy

        return PolicyRule(pattern="*", action=PolicyAction.ALLOW)
