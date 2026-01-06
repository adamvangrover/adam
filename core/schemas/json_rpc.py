from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict

class JsonRpcError(BaseModel):
    """
    Standard JSON-RPC 2.0 Error object.
    """
    code: int
    message: str
    data: Optional[Any] = None

class JsonRpcRequest(BaseModel):
    """
    Standard JSON-RPC 2.0 Request object.
    """
    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: Optional[Union[Dict[str, Any], List[Any]]] = None
    id: Optional[Union[str, int, None]] = None  # Notification if None

class JsonRpcResponse(BaseModel):
    """
    Standard JSON-RPC 2.0 Response object.
    """
    jsonrpc: Literal["2.0"] = "2.0"
    result: Optional[Any] = None
    error: Optional[JsonRpcError] = None
    id: Union[str, int, None]

    model_config = ConfigDict(validate_assignment=True)

    def is_error(self) -> bool:
        return self.error is not None

class AdaptiveConvictionMetadata(BaseModel):
    """
    Metadata for Adaptive Conviction (Metacognitive Gating).
    """
    conviction_score: float = Field(..., ge=0.0, le=1.0)
    mode: Literal["tool_execution", "elicitation", "direct_answer"]
    thought_trace: Optional[str] = None
    protocol_overhead_ms: Optional[float] = None
