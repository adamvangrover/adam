"""
Microscopic Telemetry and Logging Framework for Adam OS.
Integrates tightly with CI/CD to provide microscopic visibility into
thread execution, memory allocation, and neuro-symbolic agent decisions.
"""

import logging
import time
import os
import tempfile
from typing import Any, Dict

logger = logging.getLogger(__name__)


class MicroscopicTelemetry:
    """
    Captures extreme detail metrics for quantitative engineers to build, verify,
    and deploy complex financial algorithms with absolute confidence.
    """

    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = os.path.join(tempfile.gettempdir(), "adam_telemetry")
        self.log_dir = log_dir
        self.metrics_buffer = []
        logger.info(f"Initialized Microscopic Telemetry framework (Dir: {log_dir})")

    def record_memory_allocation(
        self, thread_id: str, size_bytes: int, pointer_addr: str
    ):
        """
        Logs a low-level memory allocation, primarily for tracking the
        zero-allocation compliance of the Iron Core during volatile markets.
        """
        self.metrics_buffer.append(
            {
                "timestamp": time.time_ns(),
                "type": "MEMORY_ALLOC",
                "thread": thread_id,
                "size": size_bytes,
                "address": pointer_addr,
            }
        )

    def record_agent_decision(
        self, agent_id: str, context: Dict[str, Any], decision: str, confidence: float
    ):
        """
        Logs the 'Narrative Logging' sequence: Event -> Analysis -> Decision -> Outcome
        used in the cognitive layer.
        """
        self.metrics_buffer.append(
            {
                "timestamp": time.time_ns(),
                "type": "AGENT_DECISION",
                "agent": agent_id,
                "context": context,
                "decision": decision,
                "confidence": confidence,
            }
        )
        logger.debug(
            f"Agent {agent_id} decided {decision} (Confidence: {confidence:.2f})"
        )

    def flush_to_cicd(self) -> int:
        """
        Simulates flushing the buffered metrics to a persistent storage or CI pipeline.
        """
        count = len(self.metrics_buffer)
        self.metrics_buffer.clear()
        logger.info(f"Flushed {count} microscopic telemetry events to CI/CD framework.")
        return count
