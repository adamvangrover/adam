"""
Example 01: Intelligence Layer Standalone
-----------------------------------------
Demonstrates running the reasoning engine (Intelligence Layer) in isolation.
Uses the ProvenanceLogger to track the decision-making process.
"""

import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.system.provenance_logger import ProvenanceLogger, ActivityType
import uuid

# Mock Agent for demonstration purposes (to avoid API key requirements)
class MockReasoningAgent:
    def __init__(self, name="Strategist-Alpha"):
        self.name = name
        self.logger = ProvenanceLogger()

    def process_query(self, query: str, context: dict) -> dict:
        trace_id = str(uuid.uuid4())
        print(f"[{self.name}] Received query: {query}")
        print(f"[{self.name}] Trace ID: {trace_id}")

        # 1. Log Receipt
        self.logger.log_activity(
            agent_id=self.name,
            activity_type=ActivityType.REVIEW,
            input_data={"query": query, "context": context},
            output_data={"status": "RECEIVED"},
            trace_id=trace_id,
            data_source="ClientAPI"
        )

        # 2. Simulate Reasoning (Chain of Thought)
        reasoning_steps = [
            "Identifying key entities...",
            "Checking risk constraints...",
            "Formulating response..."
        ]

        for step in reasoning_steps:
            print(f"[{self.name}] Thinking: {step}")
            self.logger.log_activity(
                agent_id=self.name,
                activity_type=ActivityType.GENERATION,
                input_data={"step": step},
                output_data={"status": "COMPLETED"},
                trace_id=trace_id,
                data_source="InternalModel"
            )

        # 3. Final Decision
        decision = {
            "action": "APPROVE",
            "confidence": 0.95,
            "rationale": "Query aligns with strategic mandates."
        }

        self.logger.log_activity(
            agent_id=self.name,
            activity_type=ActivityType.DECISION,
            input_data={"reasoning_trace": reasoning_steps},
            output_data=decision,
            trace_id=trace_id,
            capture_full_io=True
        )

        return decision

if __name__ == "__main__":
    print(">>> Starting Intelligence Layer Standalone Mode...")
    agent = MockReasoningAgent()

    result = agent.process_query(
        query="Authorize allocation of $5M to NVDA",
        context={"portfolio_value": 100000000}
    )

    print(f">>> Result: {result}")
    print(">>> Provenance logs written to core/libraries_and_archives/audit_trails/")
