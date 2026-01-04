from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class Metric(BaseModel):
    name: str
    value: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Span(BaseModel):
    span_id: str
    trace_id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None

class Trace(BaseModel):
    trace_id: str
    spans: List[Span]

class AgentTelemetry(BaseModel):
    agent_id: str
    metrics: List[Metric]
    traces: List[Trace]
