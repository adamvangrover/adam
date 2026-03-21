import pytest
from core.system.aof_guardrail import OversightInterventionRequired

class DummyStrategicSNCAgent:
    """
    Dummy agent meant to simulate StrategicSNCAgent behavior and gracefully throw
    OversightInterventionRequired when confidence is below threshold.
    """
    def __init__(self, name="StrategicSNCAgent"):
        self.name = name

    def execute(self, confidence: float):
        threshold = 0.85
        if confidence < threshold:
            raise OversightInterventionRequired(
                agent_name=self.name,
                reason=f"Confidence Score ({confidence:.2f}) below threshold ({threshold})",
                context={"action": "SNC execution", "confidence": confidence}
            )
        return "SNC Execution Successful"

def test_strategic_snc_agent_oversight():
    agent = DummyStrategicSNCAgent()

    # 1. Test success
    try:
        result = agent.execute(0.90)
        assert result == "SNC Execution Successful"
    except OversightInterventionRequired:
        pytest.fail("OversightInterventionRequired should not be raised on high confidence")

    # 2. Test graceful handling of OversightInterventionRequired
    try:
        agent.execute(0.50)
        pytest.fail("OversightInterventionRequired should have been raised on low confidence")
    except OversightInterventionRequired as e:
        assert e.agent_name == "StrategicSNCAgent"
        assert "Confidence Score" in e.reason
        assert "0.50" in e.reason
        assert "0.85" in e.reason
        assert e.context == {"action": "SNC execution", "confidence": 0.50}
