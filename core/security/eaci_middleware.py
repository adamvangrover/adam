import re
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EACIMiddleware:
    """
    Enterprise Adaptive Core Interface (EACI) Middleware.
    Enforces security protocols, input sanitization, and RBAC injection.
    """

    INJECTION_PATTERNS = [
        r"ignore previous instructions",
        r"system prompt",
        r"delete all files",
        r"sudo",
        r"drop table",
        r"exec\("
    ]

    def __init__(self, permission_manager=None):
        self.permission_manager = permission_manager # Placeholder for Identity Provider integration

    def sanitize_input(self, prompt: str) -> str:
        """
        Scans input for known prompt injection patterns.
        """
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                logger.warning(f"Security Alert: Injection attempt detected with pattern '{pattern}'")
                raise ValueError(f"Security Violation: Input contains forbidden pattern.")
        return prompt

    def inject_context(self, prompt: str, user_role: str) -> Dict[str, Any]:
        """
        Injects security context and permissions into the prompt envelope.
        """
        permissions = self._get_permissions(user_role)

        # In HNASP, this would update the 'meta.security_context' field
        security_context = {
            "user_role": user_role,
            "allowed_tools": permissions,
            "audit_level": "HIGH" if user_role == "GUEST" else "STANDARD"
        }

        return {
            "sanitized_prompt": prompt,
            "security_context": security_context
        }

    def _get_permissions(self, role: str) -> list:
        # Simple RBAC map
        rbac_map = {
            "ADMIN": ["*"],
            "PORTFOLIO_MANAGER": ["view_data", "run_valuation", "execute_trade"],
            "ANALYST": ["view_data", "run_valuation"],
            "GUEST": ["view_data"]
        }
        return rbac_map.get(role, [])

    def validate_tool_access(self, tool_name: str, security_context: Dict[str, Any]) -> bool:
        """
        Runtime check to see if the current context allows tool execution.
        """
        allowed = security_context.get("allowed_tools", [])
        if "*" in allowed:
            return True
        return tool_name in allowed
