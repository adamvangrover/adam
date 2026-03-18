import pytest

from core.agents.sentiment_risk_bridge import SentimentRiskBridge, SentimentRiskInput


@pytest.mark.asyncio
async def test_sentiment_risk_bridge():
    config = {"sentiment_weight": 0.5}
    agent = SentimentRiskBridge(config)

    # Test 1: High Panic (0.1 Sentiment), Low Base Risk (0.2)
    result = await agent.execute(sentiment_score=0.1, base_risk_score=0.2)
    assert result["original_risk_score"] == 0.2
    assert result["sentiment_score"] == 0.1
    # Penalty calculation: deviation = 0.1 - 0.5 = -0.4. Since < -0.2: penalty = |-0.4| * 0.5 = 0.2
    assert abs(result["applied_penalty"] - 0.2) < 1e-5
    assert abs(result["adjusted_risk_score"] - 0.4) < 1e-5

    # Test 2: High Euphoria (0.9 Sentiment), High Base Risk (0.8)
    result2 = await agent.execute(sentiment_score=0.9, base_risk_score=0.8)
    assert result2["original_risk_score"] == 0.8
    # Penalty calculation: deviation = 0.9 - 0.5 = 0.4. Since > 0.3: penalty = (0.4 * 0.5) * 0.5 = 0.1
    assert abs(result2["applied_penalty"] - 0.1) < 1e-5
    assert abs(result2["adjusted_risk_score"] - 0.9) < 1e-5

    # Test 3: Normal Conditions (0.5 Sentiment)
    result3 = await agent.execute(sentiment_score=0.5, base_risk_score=0.5)
    assert result3["applied_penalty"] == 0.0
    assert result3["adjusted_risk_score"] == 0.5

    # Test 4: Maximum cap (adjusted score should not exceed 1.0)
    result4 = await agent.execute(sentiment_score=0.0, base_risk_score=0.9)
    # Deviation = -0.5. Penalty = 0.5 * 0.5 = 0.25. Adjusted = min(1.0, 0.9 + 0.25)
    assert result4["adjusted_risk_score"] == 1.0

    # Test 5: Pydantic Input model
    input_data = SentimentRiskInput(sentiment_score=0.1, base_risk_score=0.2)
    result5 = await agent.execute(input_data)
    assert abs(result5["adjusted_risk_score"] - 0.4) < 1e-5
