from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ReasoningStep(BaseModel):
    """
    Represents a single step in the reasoning process.
    """
    step_number: int
    description: str
    action: str
    observation: str
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = {}

class TraceCollector:
    """
    Collects execution traces for agent learning and debugging.
    """
    def __init__(self):
        self.traces: List[ReasoningStep] = []

    def add_step(self, step: ReasoningStep):
        """
        Adds a reasoning step to the trace.
        """
        self.traces.append(step)
        logger.debug(f"Added reasoning step: {step.description}")

    def get_traces(self) -> List[ReasoningStep]:
        """
        Returns the collected traces.
        """
        return self.traces

    def clear(self):
        """
        Clears the collected traces.
        """
        self.traces = []

    def save_to_file(self, filepath: str):
        """
        Saves the collected traces to a JSON file.
        """
        try:
            data = [step.dict() for step in self.traces]
            # Handle datetime serialization
            for item in data:
                item['timestamp'] = item['timestamp'].isoformat()

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Traces saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving traces to file: {e}")

    def load_from_file(self, filepath: str):
        """
        Loads traces from a JSON file.
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.traces = []
            for item in data:
                item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                self.traces.append(ReasoningStep(**item))
            logger.info(f"Traces loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading traces from file: {e}")
