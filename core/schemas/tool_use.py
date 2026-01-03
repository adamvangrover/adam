from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal, Union
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict

class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None

class ToolManifest(BaseModel):
    name: str
    description: str
    version: str = "1.0.0"
    parameters: List[ToolParameter] = Field(default_factory=list)
    is_idempotent: bool = False

    model_config = ConfigDict(populate_by_name=True)

class ToolCall(BaseModel):
    call_id: str = Field(default_factory=lambda: str(uuid4()))
    tool_name: str
    arguments: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    timeout_seconds: Optional[int] = 30

class ToolResult(BaseModel):
    call_id: str
    tool_name: str
    status: Literal["success", "error", "timeout"] = "success"
    result_data: Any
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    artifacts: List[str] = Field(default_factory=list, description="Paths to any files generated")

class ToolUsageSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    calls: List[ToolCall] = Field(default_factory=list)
    results: List[ToolResult] = Field(default_factory=list)
