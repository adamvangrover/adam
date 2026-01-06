from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime

class Span(BaseModel):
    span_id: str
    trace_id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None

class Trace(BaseModel):
    trace_id: str
    spans: list[Span] = Field(default_factory=list)

class AgentTelemetry(BaseModel):
    agent_id: str
    traces: list[Trace] = Field(default_factory=list)

class Metric(BaseModel):
    name: str
    value: float
    timestamp: datetime = Field(default_factory=datetime.now)
