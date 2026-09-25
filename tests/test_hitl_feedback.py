from scripts.swarm_ledger import AppendOnlyLedger
from scripts.hitl_feedback_loop import HumanFeedback, FeedbackLoop

def test_submit_feedback():
    ledger = AppendOnlyLedger()
    loop = FeedbackLoop(ledger=ledger)
    fb = HumanFeedback(
        feedback_id="f1",
        original_message_id="m1",
        reviewer="human1",
        correction={"adj": 0.5},
        approved=True
    )
    loop.submit_feedback(fb)
    assert len(ledger.entries) == 1
    assert ledger.entries[0].sender == "human1"
    assert ledger.entries[0].payload["approved"] is True
