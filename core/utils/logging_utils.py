"""
Core Logging Utilities for ADAM OS.
Provides structured, JSON-based, and narrative logging mechanisms.
"""
import logging
import logging.config
import os
import yaml
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from contextvars import ContextVar

# Centralized Context for Trace IDs
current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)

# Handle missing pythonjsonlogger gracefully
try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False

    # Mock class to prevent import errors in type hints or basic usage
    class jsonlogger:
        class JsonFormatter(logging.Formatter):
            pass

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter ensuring UTC ISO timestamps and standard log levels.
    """
    def add_fields(self, log_record: dict[str, Any], record: logging.LogRecord, message_dict: dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.now(timezone.utc).isoformat()
        log_record['level'] = log_record.get('level', record.levelname).upper()

        # Innovator: Auto-inject trace ID if present in context
        trace_id = current_trace_id.get()
        if trace_id and 'trace_id' not in log_record:
            log_record['trace_id'] = trace_id

def setup_logging(config: dict[str, Any] | None = None, default_path: str = 'config/logging.yaml', default_level: int = logging.INFO, env_key: str = 'LOG_CFG') -> None:
    """
    Configure application logging via YAML config file or fallback to sensible defaults.
    """
    path = Path(os.getenv(env_key, default_path))

    if config:
        logging.config.dictConfig(config)
        logging.info("Logging configured from provided dictionary.")
        return

    if path.exists():
        try:
            with path.open('rt') as f:
                config_data = yaml.safe_load(f)
            logging.config.dictConfig(config_data)
            logging.info(f"Logging configured from {path}")
        except Exception as e:
            logging.basicConfig(level=default_level)
            logging.error(f"Failed to load logging config from {path}: {e}. Defaulting to basic config.")
    else:
        root_logger = logging.getLogger()
        root_logger.setLevel(default_level)
        handler = logging.StreamHandler()

        if JSON_LOGGER_AVAILABLE:
            formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
        else:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        logging.info(f"Logging configured with {'JSON' if JSON_LOGGER_AVAILABLE else 'Standard'} Formatter")

def get_logger(name: str) -> logging.Logger:
    """
    Get a standard Python logger instance.
    """
    return logging.getLogger(name)

class MilestoneLogger(logging.LoggerAdapter):
    """
    Logger adapter to visually highlight key execution milestones.
    """
    def milestone(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a milestone message as INFO level with a checkmark."""
        self.info(f"✅ Milestone: {msg}", *args, **kwargs)

def get_milestone_logger(name: str) -> MilestoneLogger:
    """
    Factory function for MilestoneLogger.
    """
    return MilestoneLogger(logging.getLogger(name), {})

class TraceLogger:
    """
    Specialized in-memory logger for recording reasoning traces and agent state transitions.
    Provides auditability and 'Thought Process' visualization capabilities.
    """
    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or current_trace_id.get() or str(uuid.uuid4())
        self.logger = logging.getLogger("TraceLogger")
        self.steps: list[dict[str, Any]] = []

    def log_step(self, agent_name: str, step_name: str, inputs: dict[str, Any], outputs: dict[str, Any], metadata: dict[str, Any] | None = None) -> None:
        """
        Record a single execution step within the current trace.
        """
        entry = {
            "trace_id": self.trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent_name,
            "step": step_name,
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata or {}
        }
        self.steps.append(entry)
        self.logger.info("Agent Step", extra=entry)

    def get_trace(self) -> list[dict[str, Any]]:
        """Retrieve the sequence of logged steps."""
        return self.steps

    def clear_trace(self) -> None:
        """Optimizer: Clear memory to prevent leaks in long-running processes."""
        self.steps.clear()

class SwarmLogger:
    """
    Singleton thread-safe structured telemetry logger for Agent Swarm execution.
    Persists JSONL events to disk for downstream analysis and UI visualization.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, log_file: str | Path = "logs/swarm_telemetry.jsonl"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SwarmLogger, cls).__new__(cls)
                cls._instance.log_file = Path(log_file)
                cls._instance.log_file.parent.mkdir(parents=True, exist_ok=True)
                cls._instance._write_lock = threading.Lock()
            return cls._instance

    def _serialize(self, obj: Any) -> Any:
        """Fallback serializer for complex objects like Pydantic models."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return str(obj)

    def log_event(self, event_type: str, agent_id: str, details: dict[str, Any]) -> None:
        """
        Record and persist a structured agent event to the telemetry log.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "agent_id": agent_id,
            "details": details,
            "trace_id": current_trace_id.get()
        }

        try:
            line = json.dumps(entry, default=self._serialize)
            with self._write_lock:
                with self.log_file.open("a") as f:
                    f.write(line + "\n")
            logging.info(f"[{event_type}] {agent_id}: {line}")
        except Exception as e:
            logging.error(f"Failed to write swarm telemetry event: {e}")

    def log_thought(self, agent_id: str, thought: str) -> None:
        """Helper to log internal agent reasoning (THOUGHT_TRACE)."""
        self.log_event("THOUGHT_TRACE", agent_id, {"content": thought})

    def log_tool(self, agent_id: str, tool_name: str, params: dict[str, Any]) -> None:
        """Helper to log tool execution (TOOL_EXECUTION)."""
        self.log_event("TOOL_EXECUTION", agent_id, {"tool": tool_name, "parameters": params})

class NarrativeLogger:
    """
    Protocol: ADAM-V-NEXT
    Logs events as a cohesive story for System 2 human-readable audits.
    Enforces 'Narrative Logging' structure: Event -> Analysis -> Decision -> Outcome.
    """
    def __init__(self, logger_name: str = "Narrative"):
        self.logger = logging.getLogger(logger_name)

    def log_narrative(self, event: str, analysis: str, decision: str, outcome: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Log a complete, structured narrative execution arc.
        """
        story = {
            "chapter": "Execution Arc",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": current_trace_id.get() or str(uuid.uuid4()),
            "1_Event": event,
            "2_Analysis": analysis,
            "3_Decision": decision,
            "4_Outcome": outcome,
            "5_Metadata": metadata or {}
        }
        self.logger.info(f"NARRATIVE:\n{json.dumps(story, indent=2)}")# Protocol Verified: ADAM-V-NEXT
