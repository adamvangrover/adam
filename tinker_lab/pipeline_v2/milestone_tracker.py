import os
import datetime
from enum import Enum
from typing import Optional

class MilestoneStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    VALIDATING = "VALIDATING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"

class MilestoneTracker:
    def __init__(self, log_dir: str = "tinker_lab/pipeline_v2/logs"):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "MILESTONES.md")
        os.makedirs(log_dir, exist_ok=True)

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write(f"# Adam SLM Training Milestones\n\n**Started:** {datetime.datetime.now()}\n\n---\n\n")

    def log_milestone(self, gate: str, status: MilestoneStatus, context: str, reasoning: Optional[str] = None):
        """
        Logs a milestone entry in the DeepMind protocol format.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"## [{timestamp}] {gate} - {status.value}\n"
        entry += f"**Context:** {context}\n"
        if reasoning:
            entry += f"**Reasoning:** {reasoning}\n"
        entry += "\n---\n\n"

        with open(self.log_file, "a") as f:
            f.write(entry)

        print(f"[MilestoneTracker] {gate}: {status.value} - {context}")

    def start_gate(self, gate: str, description: str):
        self.log_milestone(gate, MilestoneStatus.IN_PROGRESS, description)

    def complete_gate(self, gate: str, result_summary: str):
        self.log_milestone(gate, MilestoneStatus.COMPLETE, result_summary, "Conditions met. Proceeding.")

    def fail_gate(self, gate: str, error: str):
        self.log_milestone(gate, MilestoneStatus.FAILED, f"Gate failed due to error: {error}", "Stopping pipeline.")
