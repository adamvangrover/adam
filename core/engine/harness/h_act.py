"""h_act — post-tool hook: truncate stdout and normalize errors into a ``ToolResult``."""
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel

from core.engine.harness.policy import ActionPolicy


class ToolResult(BaseModel):
    ok: bool
    stdout: str = ""
    truncated: bool = False
    error: Optional[Dict[str, str]] = None
    data: Any = None


def truncate(text: str, policy: ActionPolicy) -> Tuple[str, bool]:
    if len(text) <= policy.max_stdout_chars:
        return text, False
    omitted = len(text) - policy.max_stdout_chars
    return text[:policy.max_stdout_chars] + policy.truncation_marker.format(omitted=omitted), True


def normalize_error(error: Any, policy: ActionPolicy) -> Dict[str, str]:
    """Map an exception, ``{"error": ...}`` dict or string onto ``{"type", "message"}``."""
    if isinstance(error, BaseException):
        kind, message = type(error).__name__, str(error)
    elif isinstance(error, dict):
        kind, message = str(error.get("error_type", "ToolError")), str(error.get("error", error))
    else:
        kind, message = "ToolError", str(error)
    return {"type": kind, "message": truncate(message, policy)[0]}


def post_tool(raw: Any, policy: ActionPolicy) -> ToolResult:
    if isinstance(raw, BaseException) or (isinstance(raw, dict) and raw.get("error")):
        return ToolResult(ok=False, error=normalize_error(raw, policy),
                          data=raw if isinstance(raw, dict) else None)
    if isinstance(raw, str):
        stdout, cut = truncate(raw, policy)
        return ToolResult(ok=True, stdout=stdout, truncated=cut)
    if isinstance(raw, dict) and isinstance(raw.get("stdout"), str):
        stdout, cut = truncate(raw["stdout"], policy)
        return ToolResult(ok=True, stdout=stdout, truncated=cut, data={**raw, "stdout": stdout})
    return ToolResult(ok=True, data=raw)
