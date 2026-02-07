import json
import uuid
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class TraceLog:
    """
    Layer 4: HITL Dashboard & Tracing.
    Logs every decision, input, and evaluation for audit trails.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.trace_id = str(uuid.uuid4())
        self.session_id = session_id or "default_session"
        self.steps: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

    def log_event(self,
                  component: str,
                  event_type: str,
                  payload: Any,
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Logs a specific event in the trace.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "event_type": event_type,
            "payload": payload,
            "metadata": metadata or {}
        }
        self.steps.append(entry)
        logger.info(f"[{component}] {event_type}")

    def add_confidence_score(self, output_payload: Dict[str, Any], evaluation_score: float) -> Dict[str, Any]:
        """
        Enhances an output payload with a calculated Confidence Interval.
        """
        # Heuristic: Confidence scales with the evaluation score (normalized 0-1)
        # In a real system, this would come from the model logits or ensemble agreement.
        base_confidence = 0.85
        if evaluation_score > 4.5:
            confidence = 0.98
        elif evaluation_score > 3.5:
            confidence = 0.92
        elif evaluation_score < 2.0:
            confidence = 0.40
        else:
            confidence = 0.75

        output_payload["_meta"] = output_payload.get("_meta", {})
        output_payload["_meta"]["confidence_score"] = confidence
        output_payload["_meta"]["confidence_rationale"] = f"Based on Auditor Score of {evaluation_score}/5.0"

        return output_payload

    def save_trace(self, filepath: str = "traces.jsonl"):
        """
        Persists the trace to a JSONL file.
        """
        record = {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "step_count": len(self.steps),
            "events": self.steps
        }

        try:
            with open(filepath, 'a') as f:
                f.write(json.dumps(record) + "\n")
            logger.info(f"Trace {self.trace_id} saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save trace: {e}")

    def get_summary(self) -> str:
        return f"Trace {self.trace_id}: {len(self.steps)} events recorded."
