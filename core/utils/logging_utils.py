# Verified for Adam v25.5
# Reviewed by Jules
import logging
import logging.config
import os
import yaml
import json
from datetime import datetime, timezone
import uuid
from typing import Dict, Any, Optional

# Handle missing pythonjsonlogger for lightweight environments
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
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.now(timezone.utc).isoformat()
        log_record['level'] = log_record.get('level', record.levelname).upper()

def setup_logging(config: Optional[Dict[str, Any]] = None, default_path: str = 'config/logging.yaml', default_level: int = logging.INFO, env_key: str = 'LOG_CFG') -> None:
    """
    Configure application logging via YAML config file or fallback to sensible defaults.
    """
    path = os.getenv(env_key, default_path)

    if os.path.exists(path):
        try:
            with open(path, 'rt') as f:
                config_data = yaml.safe_load(f)
                logging.config.dictConfig(config_data)
                logging.info(f"Logging configured from {path}")
        except Exception as e:
            print(f"Error loading logging config from {path}: {e}")
            logging.basicConfig(level=default_level)
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
    def milestone(self, msg: str, *args, **kwargs) -> None:
        """Log a milestone message as INFO level with a checkmark."""
        self.info(f"✅ Milestone: {msg}", *args, **kwargs)

def get_milestone_logger(name: str) -> MilestoneLogger:
    """
    Factory function for MilestoneLogger.
    """
    return MilestoneLogger(logging.getLogger(name), {})

class TraceLogger:
    """
    Specialized logger for recording agent reasoning traces and state transitions.
    Provides auditability and 'Thought Process' visualization capabilities.
    """
    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.logger = logging.getLogger("TraceLogger")
        self.steps = []

    def log_step(self, agent_name: str, step_name: str, inputs: Dict[str, Any], outputs: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
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

    def get_trace(self) -> list:
        """Retrieve the sequence of logged steps."""
        return self.steps

class SwarmLogger:
    """
    Singleton structured telemetry logger for Agent Swarm execution.
    Persists JSONL events to disk for analysis and UI visualization.
    """
    _instance = None

    def __new__(cls, log_file: str = "logs/swarm_telemetry.jsonl"):
        if cls._instance is None:
            cls._instance = super(SwarmLogger, cls).__new__(cls)
            cls._instance.log_file = log_file
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        return cls._instance

    def _default_serializer(self, obj: Any) -> Any:
        """Fallback serializer for complex objects like Pydantic models."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return str(obj)

    def log_event(self, event_type: str, agent_id: str, details: Dict[str, Any]) -> None:
        """
        Record and persist a structured agent event to the telemetry log.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "agent_id": agent_id,
            "details": details
        }

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry, default=self._default_serializer) + "\n")

            logging.info(f"[{event_type}] {agent_id}: {json.dumps(details, default=self._default_serializer)}")
        except Exception as e:
            logging.error(f"Failed to write swarm telemetry event: {e}")

    def log_thought(self, agent_id: str, thought: str) -> None:
        """Helper to log internal agent reasoning (THOUGHT_TRACE)."""
        self.log_event("THOUGHT_TRACE", agent_id, {"content": thought})

    def log_tool(self, agent_id: str, tool_name: str, params: Dict[str, Any]) -> None:
        """Helper to log tool execution (TOOL_EXECUTION)."""
        self.log_event("TOOL_EXECUTION", agent_id, {"tool": tool_name, "parameters": params})

class NarrativeLogger:
    """
    Protocol: ADAM-V-NEXT
    Enforces 'Narrative Logging' structure: Event -> Analysis -> Decision -> Outcome.
    """
    def __init__(self, logger_name: str = "Narrative"):
        self.logger = logging.getLogger(logger_name)

    def log_narrative(self, event: str, analysis: str, decision: str, outcome: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a complete, structured narrative execution arc.
        """
        story = {
            "chapter": "Execution Arc",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": str(uuid.uuid4()),
            "Event": event,
            "Analysis": analysis,
            "Decision": decision,
            "Outcome": outcome,
            "Metadata": metadata or {}
        }
        self.logger.info(f"NARRATIVE:\n{json.dumps(story, indent=2)}")
