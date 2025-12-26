import logging
import logging.config
import os
import yaml
import json
from datetime import datetime
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
        logging.basicConfig(level=default_level)
        logging.info("Logging configured with default basicConfig")


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

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            # Fallback to standard logging if file write fails
            logging.error(f"Failed to write swarm telemetry: {e}")

    def log_thought(self, agent_id: str, thought: str):
        self.log_event("THOUGHT_TRACE", agent_id, {"content": thought})

    def log_tool(self, agent_id: str, tool_name: str, params: Dict[str, Any]):
        self.log_event("TOOL_EXECUTION", agent_id, {"tool": tool_name, "parameters": params})
