# Verified for Adam v25.5
# Reviewed by Jules
import logging
import logging.config
import os
import yaml
import json
from datetime import datetime
import uuid

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
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            # this doesn't use record.created, so it is slightly off
            now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            log_record['timestamp'] = now
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname

from typing import Dict, Any, Optional

def setup_logging(config=None, default_path='config/logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """
    Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value

    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                logging.info(f"Logging configured from {path}")
            except Exception as e:
                print(f"Error loading logging config: {e}")
                logging.basicConfig(level=default_level)
    else:
        # Default to JSON logging if no config file found (Modern default)
        root_logger = logging.getLogger()
        root_logger.setLevel(default_level)

        handler = logging.StreamHandler()

        if JSON_LOGGER_AVAILABLE:
            formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
            handler.setFormatter(formatter)
        else:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)

        root_logger.addHandler(handler)

        logging.info(f"Logging configured with {'JSON' if JSON_LOGGER_AVAILABLE else 'Standard'} Formatter")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    """
    return logging.getLogger(name)

class MilestoneLogger(logging.LoggerAdapter):
    """
    Logger adapter to enforce consistent milestone logging.
    """
    def milestone(self, msg, *args, **kwargs):
        self.info(f"✅ Milestone: {msg}", *args, **kwargs)

def get_milestone_logger(name: str) -> MilestoneLogger:
    """
    Get a MilestoneLogger instance.
    """
    logger = logging.getLogger(name)
    return MilestoneLogger(logger, {})

class TraceLogger:
    """
    Specialized logger for recording reasoning traces and agent state transitions.
    Useful for auditability and 'Thought Process' visualization.
    """
    def __init__(self, trace_id: str = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.logger = logging.getLogger("TraceLogger")
        self.steps = []

    def log_step(self, agent_name: str, step_name: str, inputs: dict, outputs: dict, metadata: dict = None):
        entry = {
            "trace_id": self.trace_id,
            "timestamp": datetime.utcnow().isoformat(),
            "agent": agent_name,
            "step": step_name,
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata or {}
        }
        self.steps.append(entry)
        # Log as structured JSON
        self.logger.info("Agent Step", extra=entry)

    def get_trace(self):
        return self.steps

class SwarmLogger:
    """
    Structured telemetry logger for the Agent Swarm.
    Writes JSONL events to a persistent log file for analysis and UI visualization.
    """
    _instance = None

    def __new__(cls, log_file: str = "logs/swarm_telemetry.jsonl"):
        if cls._instance is None:
            cls._instance = super(SwarmLogger, cls).__new__(cls)
            cls._instance.log_file = log_file
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        return cls._instance

    def log_event(self, event_type: str, agent_id: str, details: Dict[str, Any]):
        """
        Log a structured event.

        Args:
            event_type (str): e.g., "TASK_START", "TOOL_USE", "CRITIQUE", "ERROR"
            agent_id (str): The name of the agent generating the event.
            details (Dict[str, Any]): Payload of the event.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "agent_id": agent_id,
            "details": details
        }

        def default_serializer(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "dict"):
                return obj.dict()
            return str(obj)

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry, default=default_serializer) + "\n")

            # Also log to standard logging for visibility
            logging.info(f"[{event_type}] {agent_id}: {json.dumps(details, default=default_serializer)}")

        except Exception as e:
            # Fallback to standard logging if file write fails
            logging.error(f"Failed to write swarm telemetry: {e}")

    def log_thought(self, agent_id: str, thought: str):
        self.log_event("THOUGHT_TRACE", agent_id, {"content": thought})

    def log_tool(self, agent_id: str, tool_name: str, params: Dict[str, Any]):
        self.log_event("TOOL_EXECUTION", agent_id, {"tool": tool_name, "parameters": params})


class NarrativeLogger:
    """
    Protocol: ADAM-V-NEXT
    Logs events as a cohesive story: Event -> Analysis -> Decision -> Outcome.
    Protocol Verified: ADAM-V-NEXT (Updated)
    Verified structured narrative output format.
    """
    def __init__(self, logger_name: str = "Narrative"):
        self.logger = logging.getLogger(logger_name)

    def log_narrative(self, event: str, analysis: str, decision: str, outcome: str):
        """
        Log a complete narrative arc.
        """
        story = {
            "chapter": "Execution Arc",
            "1_Event": event,
            "2_Analysis": analysis,
            "3_Decision": decision,
            "4_Outcome": outcome
        }
        self.logger.info(f"NARRATIVE: {json.dumps(story, indent=2)}")
