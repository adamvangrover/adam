from pydantic import BaseModel, ConfigDict
from scripts.swarm_ledger import SwarmMessage, AppendOnlyLedger

class HumanFeedback(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    feedback_id: str
    original_message_id: str
    reviewer: str
    correction: dict
    approved: bool

class FeedbackLoop(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    ledger: AppendOnlyLedger

    def submit_feedback(self, feedback: HumanFeedback):
        msg = SwarmMessage(
            message_id=f"fb-{feedback.feedback_id}",
            sender=feedback.reviewer,
            payload=feedback.model_dump(),
            timestamp=1000  # Mock timestamp
        )
        self.ledger.append(msg)
