from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import uuid
import contextlib

class Metric(BaseModel):
    name: str
    value: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = Field(default_factory=dict)

class Span(BaseModel):
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    name: str
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: str = "IN_PROGRESS" # SUCCESS, ERROR

    def end(self, status: str = "SUCCESS"):
        self.end_time = datetime.now(timezone.utc)
        self.status = status

    @property
    def duration_ms(self) -> float:
        if not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000

class Trace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    spans: List[Span] = Field(default_factory=list)
    root_span_id: Optional[str] = None

    def add_span(self, span: Span):
        self.spans.append(span)
        if not self.root_span_id and not span.parent_id:
            self.root_span_id = span.span_id

class AgentTelemetry(BaseModel):
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
        Context manager to create and manage a span.
        """
        if not self.active_trace:
            self.start_trace()

        # Auto-detect parent from context if possible, or use provided
        # For this simple implementation, we rely on parent_id passed explicitly
        # or just attach to root if it's the first span.

        span = Span(name=name, parent_id=parent_id)
        self.active_trace.add_span(span)

        try:
            yield span
            span.end(status="SUCCESS")
        except Exception as e:
            span.end(status="ERROR")
            span.metadata["error"] = str(e)
            raise e
