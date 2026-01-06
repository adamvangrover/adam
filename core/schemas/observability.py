from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class Metric(BaseModel):
    """
    Represents a single quantitative data point.
    Merged: Uses 'utcnow' (main) for telemetry standardization.
    """
    name: str
    value: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Span(BaseModel):
    """
    Represents a single operation within a trace.
    """
    span_id: str
    trace_id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None

class Trace(BaseModel):
    """
    Represents a collection of spans describing a workflow.
    Merged: Uses List typing (main) with default factories (v24).
    """
    trace_id: str
    spans: List[Span] = Field(default_factory=list)

class AgentTelemetry(BaseModel):
    """
    Aggregated telemetry data for a specific agent.
    Merged: Includes 'metrics' (main) and 'traces' (both), with 
    robust default initialization (v24).
    """
    agent_id: str
    metrics: List[Metric] = Field(default_factory=list)
    traces: List[Trace] = Field(default_factory=list)