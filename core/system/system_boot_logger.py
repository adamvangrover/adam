import json
import time
from dataclasses import dataclass, asdict
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class BootLogEntry:
    timestamp: float
    agent_id: str
    status: str
    highest_conviction_prompt: str
    conviction_score: float
    version: str = "v30.0-alpha"  # Default system version

class SystemBootLogger:
    """
    Handles logging of system boot events and agent conviction states
    to a version control log file.
    """
    LOG_FILE = "logs/version_control_log.jsonl"

    @classmethod
    def log_boot(cls, entry: BootLogEntry):
        try:
            # Ensure log directory exists
            os.makedirs(os.path.dirname(cls.LOG_FILE), exist_ok=True)

            with open(cls.LOG_FILE, "a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")

            logger.info(f"Boot logged for {entry.agent_id}: {entry.status}")
        except Exception as e:
            logger.error(f"Failed to log boot entry: {e}")
