"""
Agent Notes:
    - Role: Principal AI Architect
    - Module: Trace Collector & Artisanal Data Synthesizer
    - Purpose: Captures agent execution traces (Reasoning -> Action -> Outcome)
      and serializes them into:
        1. High-quality training datasets (JSONL) for DPO/SFT.
        2. Raw debug logs (JSON) for immediate introspection.
    - Philosophy: Every interaction is a training example. We move from 'ephemeral'
      compute to 'persistent' knowledge.
"""

import json
import time
import uuid
import os
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

# --- Configuration & Setup ---
DATA_DIR = os.path.join("data", "artisanal_training_sets")
DEBUG_DIR = os.path.join("data", "debug_traces")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TraceCollector")


# --- Enums & Data Models ---

class TraceType(str, Enum):
    """Classifies the final outcome for Dataset sorting (DPO pairs)."""
    SUCCESS = "success"
    FAILURE = "failure"
    CORRECTION = "correction"  # High value: Agent fixed its own mistake


class StepType(str, Enum):
    """Standardizes the atomic units of agent thought."""
    THOUGHT = "thought"
    TOOL_USE = "tool_use"
    OBSERVATION = "observation"
    OUTPUT = "output"
    SYSTEM = "system"


class ReasoningStep(BaseModel):
    """
    Represents a single atomic step in the agent's reasoning loop.
    Uses Pydantic for strict validation.
    """
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    iso_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    agent_id: str
    step_type: StepType
    content: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class AgentTrace:
    """
    Container for a full session or task execution.
    Manages the narrative arc of the agent's work.
    """
    def __init__(self, trace_id: Optional[str] = None, task_description: str = ""):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.task_description = task_description
        self.steps: List[ReasoningStep] = []
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.final_outcome: Optional[str] = None
        self.trace_type: TraceType = TraceType.SUCCESS
        self.feedback_score: float = 0.0

    def add_step(self, step: ReasoningStep):
        self.steps.append(step)

    def close(self, outcome: str, success: bool = True, score: float = 1.0):
        self.final_outcome = outcome
        self.end_time = time.time()
        self.trace_type = TraceType.SUCCESS if success else TraceType.FAILURE
        self.feedback_score = score

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the full object for debug storage."""
        return {
            "trace_id": self.trace_id,
            "task": self.task_description,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": (self.end_time or time.time()) - self.start_time,
            "outcome": self.final_outcome,
            "type": self.trace_type,
            "score": self.feedback_score,
            "steps": [step.dict() for step in self.steps]
        }


# --- Main Collector ---

class TraceCollector:
    """
    Global manager for collecting, debugging, and synthesizing traces.
    """
    def __init__(self):
        self.active_traces: Dict[str, AgentTrace] = {}

    def start_trace(self, task_description: str) -> str:
        """Starts a new trace session and returns its ID."""
        trace = AgentTrace(task_description=task_description)
        self.active_traces[trace.trace_id] = trace
        logger.info(f"Started trace {trace.trace_id} for task: '{task_description}'")
        return trace.trace_id

    def log(self, trace_id: str, agent_id: str, step_type: Union[StepType, str], content: Any, meta: Dict = None):
        """
        Logs a specific step to an active trace.
        """
        if trace_id not in self.active_traces:
            logger.warning(f"Attempted to log to non-existent trace: {trace_id}")
            return

        # Ensure Enum consistency
        if isinstance(step_type, str):
            try:
                step_type = StepType(step_type)
            except ValueError:
                logger.warning(f"Invalid StepType '{step_type}', defaulting to THOUGHT")
                step_type = StepType.THOUGHT

        step = ReasoningStep(
            agent_id=agent_id,
            step_type=step_type,
            content=content,
            metadata=meta or {}
        )
        
        self.active_traces[trace_id].add_step(step)
        logger.debug(f"[{agent_id}] {step_type.value}: {str(content)[:50]}...")

    def end_trace(self, trace_id: str, outcome: str, success: bool = True, score: float = 1.0):
        """
        Closes a trace and triggers the dual-export process (Debug + Training Data).
        """
        if trace_id not in self.active_traces:
            logger.error(f"Cannot end unknown trace: {trace_id}")
            return

        trace = self.active_traces[trace_id]
        trace.close(outcome, success, score)

        # 1. Save Raw Debug Log (JSON) - Good for humans/debugging
        self._save_debug_log(trace)

        # 2. Export Artisanal Training Data (JSONL) - Good for LLMs
        self._export_to_training_set(trace)

        # Cleanup memory
        del self.active_traces[trace_id]
        logger.info(f"Trace {trace_id} completed. Outcome: {trace.trace_type}")

    def _save_debug_log(self, trace: AgentTrace):
        """Dumps the raw trace state to a JSON file."""
        filename = f"trace_{trace.trace_id}.json"
        filepath = os.path.join(DEBUG_DIR, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(trace.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save debug log: {e}")

    def _export_to_training_set(self, trace: AgentTrace):
        """
        Synthesizes the trace into a ChatML/ShareGPT format for LLM training.
        Applies specific control tokens for thoughts/tools.
        """
        # 1. System Prompt
        messages = [
            {"role": "system", "content": "You are Adam, an advanced AI financial architect."}
        ]
        
        # 2. User Prompt
        messages.append({"role": "user", "content": trace.task_description})

        # 3. Compress steps into a coherent assistant response chain
        # NOTE: This logic flattens the reasoning into a single turn for SFT.
        thought_chain = ""
        for step in trace.steps:
            if step.step_type == StepType.THOUGHT:
                # Custom control tokens for internal monologue
                thought_chain += f"<\ctrl3347>{step.content}<\ctrl3348>\n"
            elif step.step_type == StepType.TOOL_USE:
                thought_chain += f"<tool>{step.content}</tool>\n"
            elif step.step_type == StepType.OBSERVATION:
                thought_chain += f"<observation>{step.content}</observation>\n"
            elif step.step_type == StepType.OUTPUT:
                thought_chain += f"{step.content}\n"

        messages.append({"role": "assistant", "content": thought_chain.strip()})

        entry = {
            "id": trace.trace_id,
            "timestamp": datetime.fromtimestamp(trace.start_time).isoformat(),
            "score": trace.feedback_score,
            "messages": messages
        }

        # 4. Determine destination based on quality
        if trace.trace_type == TraceType.SUCCESS:
            filename = "adam_finetune_success.jsonl"
        else:
            filename = "adam_finetune_failures.jsonl"
            
        filepath = os.path.join(DATA_DIR, filename)

        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            logger.info(f"Appended training example to {filename}")
        except Exception as e:
            logger.critical(f"Failed to save training data: {e}")


# --- Example Usage ---

if __name__ == "__main__":
    # Initialize
    collector = TraceCollector()
    
    # Start a session
    tid = collector.start_trace("Analyze the credit risk of AAPL")
    
    # Simulate Agent Activity
    collector.log(tid, "RiskAgent", "thought", "Retrieving balance sheet data to check liquidity.")
    
    # Simulate Tool Use
    tool_call = "get_balance_sheet(ticker='AAPL')"
    collector.log(tid, "RiskAgent", "tool_use", tool_call)
    
    # Simulate Observation (New feature from Snippet 2)
    obs = {"current_ratio": 1.2, "debt_to_equity": 1.5}
    collector.log(tid, "RiskAgent", "observation", json.dumps(obs))
    
    # Simulate Reasoning
    collector.log(tid, "RiskAgent", "thought", "Debt to Equity ratio looks healthy. Proceeding with low risk assessment.")
    
    # Final Output
    collector.log(tid, "RiskAgent", "output", "AAPL Credit Risk is LOW based on liquidity metrics.")
    
    # End Trace
    collector.end_trace(tid, "Low Risk", success=True, score=0.98)
    
    print(f"Check {DATA_DIR} for JSONL files and {DEBUG_DIR} for raw traces.")
