from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import uuid
import contextlib

class Metric(BaseModel):
    """
    Represents a single quantitative data point.
    Merged: Adopted 'datetime.now(timezone.utc)' (guide) over 'utcnow' (main)
    as 'utcnow' is deprecated in modern Python. Retained 'tags' for dimensionality.
    """
    name: str
    value: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = Field(default_factory=dict)

class Span(BaseModel):
    """
    Represents a single operation within a trace.
    Merged: Combines 'trace_id' (main) for flat-log correlation with 
    'parent_id'/'status' (guide) for hierarchical tree reconstruction.
    """
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str  # Explicit link to the parent trace (from main)
    parent_id: Optional[str] = None
    name: str
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: str = "IN_PROGRESS"  # SUCCESS, ERROR

    def end(self, status: str = "SUCCESS"):
        """Finalizes the span with a timestamp and status."""
        self.end_time = datetime.now(timezone.utc)
        self.status = status

    @property
    def duration_ms(self) -> float:
        """Calculates duration in milliseconds."""
        if not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000

class Trace(BaseModel):
    """
    Represents a collection of spans describing a workflow.
    """
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    spans: List[Span] = Field(default_factory=list)
    root_span_id: Optional[str] = None

    def add_span(self, span: Span):
        self.spans.append(span)
        if not self.root_span_id and not span.parent_id:
            self.root_span_id = span.span_id

class AgentTelemetry(BaseModel):
    """
    Aggregated telemetry data for a specific agent.
    Merged: Retained the context manager (guide) as it significantly reduces
    boilerplate code when instrumenting agents.
    """
    agent_id: str
    metrics: List[Metric] = Field(default_factory=list)
    traces: List[Trace] = Field(default_factory=list)
    active_trace: Optional[Trace] = None

    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        metric = Metric(name=name, value=value, tags=tags or {})
        self.metrics.append(metric)

    def start_trace(self) -> Trace:
        trace = Trace()
        self.traces.append(trace)
        self.active_trace = trace
        return trace

    @contextlib.contextmanager
    def span(self, name: str, parent_id: Optional[str] = None):
        """
        Context manager to create, track, and close a span automatically.
        """
        if not self.active_trace:
            self.start_trace()

        # Ensure trace_id is passed down to the Span
        current_trace_id = self.active_trace.trace_id

        span = Span(
            name=name, 
            trace_id=current_trace_id, 
            parent_id=parent_id
        )
        self.active_trace.add_span(span)

        try:
            yield span
            span.end(status="SUCCESS")
        except Exception as e:
            span.end(status="ERROR")
            span.metadata["error"] = str(e)
            raise e