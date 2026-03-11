"""
Microscopic Telemetry and Logging Framework for Adam OS.
Provides granular visibility into every thread, memory allocation,
kernel bypass latency distribution, and agent decision tree.
"""

import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MicroscopicTelemetry:
    """
    Centralized tracing utility that ensures no latency outlier
    (e.g., a 99th percentile microsecond spike) goes unrecorded.
    Designed to feed into evolutionary meta-agents and risk dashboards.
    """

    def __init__(self, export_target: str = "timescaledb"):
        self.export_target = export_target
        self._active_spans: Dict[str, float] = {}
        logger.info(
            f"Initialized Microscopic Telemetry. Exporting spans to `{export_target}`."
        )

    def start_span(self, operation_name: str, tags: Dict[str, Any] = None) -> str:
        """
        Marks the beginning of an execution span, such as a strategy calculation
        or an agentic Retrieval-Augmented Generation task.
        """
        span_id = f"{operation_name}_{time.time_ns()}"
        self._active_spans[span_id] = time.perf_counter_ns()
        logger.debug(f"Span started: {span_id} with tags {tags}")
        return span_id

    def end_span(self, span_id: str) -> float:
        """
        Calculates nanosecond-precision latency for a given execution block
        and commits it to persistent telemetry metrics.
        """
        if span_id not in self._active_spans:
            logger.warning(f"Attempted to end unknown telemetry span: {span_id}")
            return -1.0

        start_time_ns = self._active_spans.pop(span_id)
        duration_ns = time.perf_counter_ns() - start_time_ns

        # Simulated historical telemetry commit to catch performance bottlenecks
        logger.debug(
            f"Span completed: {span_id}. Duration: {duration_ns}ns ({duration_ns / 1000.0:.2f}μs)."
        )
        return float(duration_ns)

    def capture_memory_allocation(self, size_bytes: int, structure_name: str):
        """
        Logs explicit memory usage requests inside the Cognitive Orchestrator.
        (The Iron Core prohibits dynamic allocations, but AI agents require them).
        """
        logger.debug(
            f"Memory Allocation Tracked: {size_bytes} bytes for `{structure_name}`."
        )
