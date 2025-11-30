"""
Agent Notes:
    - Role: Principal AI Architect
    - Module: Trace Collector & Artisanal Data Synthesizer
    - Purpose: Captures agent execution traces (Reasoning -> Action -> Outcome)
      and serializes them into high-quality training datasets (JSONL).
    - Philosophy: Every interaction is a training example. We move from 'ephemeral'
      compute to 'persistent' knowledge.
    - Format: Supports ShareGPT and Standard Instruction formats for DPO/SFT.
"""

import json
import time
import uuid
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

# Configuration
DATA_DIR = os.path.join("data", "artisanal_training_sets")
os.makedirs(DATA_DIR, exist_ok=True)

class TraceType(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CORRECTION = "correction"  # Valuable for DPO (Negative -> Positive)

class ReasoningStep(BaseModel):
    timestamp: float
    agent_id: str
    step_type: str # 'thought', 'tool_use', 'output'
    content: Any
    metadata: Dict[str, Any] = {}

class AgentTrace:
    """Represents a full session or task execution."""
    def __init__(self, trace_id: Optional[str] = None, task_description: str = ""):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.task_description = task_description
        self.steps: List[ReasoningStep] = []
        self.start_time = time.time()
        self.final_outcome: Optional[str] = None
        self.trace_type: TraceType = TraceType.SUCCESS
        self.feedback_score: float = 0.0

    def add_step(self, agent_id: str, step_type: str, content: Any, meta: Dict = {}):
        self.steps.append(ReasoningStep(
            timestamp=time.time(),
            agent_id=agent_id,
            step_type=step_type,
            content=content,
            metadata=meta
        ))

    def close(self, outcome: str, success: bool = True, score: float = 1.0):
        self.final_outcome = outcome
        self.trace_type = TraceType.SUCCESS if success else TraceType.FAILURE
        self.feedback_score = score

class TraceCollector:
    """
    Global singleton-like class to manage the collection and saving of traces.
    """
    def __init__(self):
        self.active_traces: Dict[str, AgentTrace] = {}

    def start_trace(self, task_description: str) -> str:
        """Starts a new trace and returns its ID."""
        trace = AgentTrace(task_description=task_description)
        self.active_traces[trace.trace_id] = trace
        return trace.trace_id

    def log(self, trace_id: str, agent_id: str, step_type: str, content: Any, meta: Dict = {}):
        if trace_id in self.active_traces:
            self.active_traces[trace_id].add_step(agent_id, step_type, content, meta)

    def end_trace(self, trace_id: str, outcome: str, success: bool = True, score: float = 1.0):
        if trace_id in self.active_traces:
            trace = self.active_traces[trace_id]
            trace.close(outcome, success, score)
            self._export_trace(trace)
            del self.active_traces[trace_id]

    def _export_trace(self, trace: AgentTrace):
        """Exports the trace to the appropriate JSONL file based on type."""
        
        # Construct the conversational format
        messages = [{"role": "system", "content": "You are Adam, an advanced AI financial architect."}]
        messages.append({"role": "user", "content": trace.task_description})
        
        # Compress steps into a coherent assistant response chain
        # This logic can be expanded to support multi-turn. For now, we flatten.
        thought_chain = ""
        for step in trace.steps:
            if step.step_type == 'thought':
                thought_chain += f"<ctrl3347>{step.content}<ctrl3348>\n"
            elif step.step_type == 'tool_use':
                thought_chain += f"<tool>{step.content}</tool>\n"
            elif step.step_type == 'output':
                thought_chain += f"{step.content}\n"

        messages.append({"role": "assistant", "content": thought_chain.strip()})

        entry = {
            "id": trace.trace_id,
            "timestamp": datetime.fromtimestamp(trace.start_time).isoformat(),
            "score": trace.feedback_score,
            "messages": messages
        }

        # Determine filename based on task type or success
        filename = "adam_finetune_success.jsonl" if trace.trace_type == TraceType.SUCCESS else "adam_finetune_failures.jsonl"
        filepath = os.path.join(DATA_DIR, filename)

        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"CRITICAL: Failed to save training data: {e}")

# Example Usage
if __name__ == "__main__":
    collector = TraceCollector()
    tid = collector.start_trace("Analyze the credit risk of AAPL")
    collector.log(tid, "RiskAgent", "thought", "Retrieving balance sheet data.")
    collector.log(tid, "RiskAgent", "tool_use", "get_balance_sheet(ticker='AAPL')")
    collector.log(tid, "RiskAgent", "thought", "Debt to Equity ratio looks healthy at 1.5.")
    collector.log(tid, "RiskAgent", "output", "AAPL Credit Risk is LOW based on liquidity metrics.")
    collector.end_trace(tid, "Low Risk", success=True, score=0.95)
    print("Trace collected and saved.")
