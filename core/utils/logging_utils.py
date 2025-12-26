import logging
import logging.config
import os
import yaml
import json
from datetime import datetime
import uuid
from pythonjsonlogger import jsonlogger

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
        handler = logging.StreamHandler()
        formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
        handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(default_level)

        # Prevent duplicate logs if basicConfig was already called
        if not root_logger.handlers:
            logging.basicConfig(level=default_level)

        logging.info("Logging configured with default JSON Formatter")


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
