**1. JULES' RATIONALE:**
> "I noticed we have `RiskAssessmentAgent` and `MarketSentimentAgent`, but no component bridging these two. `MarketSentimentAgent` provides an overall market sentiment score and details. `RiskAssessmentAgent` accepts market data which contains price data to calculate VaR, but doesn't take raw sentiment to adjust baseline risks. I researched and built `SentimentRiskBridge` to map market sentiment directly to risk, adding risk penalties for extreme panic or euphoria."

**2. FILE: core/agents/sentiment_risk_bridge.py**
```python
from typing import Any, Dict, Optional
import logging
from core.agents.agent_base import AgentBase
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class SentimentRiskInput(BaseModel):
    sentiment_score: float
    base_risk_score: float

class SentimentRiskBridge(AgentBase):
    """
    Bridge component that correlates MarketSentiment output with RiskAssessment output.
    Adjusts the baseline risk score based on extreme sentiment readings (e.g. panic or euphoria).
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Any] = None):
        super().__init__(config, kernel=kernel)
        self.sentiment_weight = self.config.get("sentiment_weight", 0.3)

    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Executes the risk adjustment logic.

        Accepts either kwargs or an instance of SentimentRiskInput via *args.
        """
        sentiment_score = 0.5
        base_risk_score = 0.5

        if "sentiment_score" in kwargs and "base_risk_score" in kwargs:
            sentiment_score = kwargs["sentiment_score"]
            base_risk_score = kwargs["base_risk_score"]
        elif args and hasattr(args[0], "sentiment_score") and hasattr(args[0], "base_risk_score"):
            sentiment_score = args[0].sentiment_score
            base_risk_score = args[0].base_risk_score
        else:
             logger.warning("SentimentRiskBridge: Missing sentiment or risk score. Proceeding with defaults.")

        # Sentiment is 0.0 (Bearish/Panic) to 1.0 (Bullish/Euphoria). Center is 0.5.
        # Base Risk is 0.0 (Low Risk) to 1.0 (High Risk).

        # We increase risk if sentiment is heavily bearish (panic sell-off).
        # We also slightly increase risk if sentiment is heavily bullish (bubble/euphoria).

        sentiment_deviation = sentiment_score - 0.5

        # Risk penalty function
        if sentiment_deviation < -0.2:
            # High panic, high risk penalty
            penalty = abs(sentiment_deviation) * self.sentiment_weight
        elif sentiment_deviation > 0.3:
            # High euphoria, moderate risk penalty
            penalty = (sentiment_deviation * 0.5) * self.sentiment_weight
        else:
            # Normal market conditions, no significant penalty
            penalty = 0.0

        adjusted_risk_score = min(1.0, base_risk_score + penalty)

        return {
            "original_risk_score": base_risk_score,
            "sentiment_score": sentiment_score,
            "applied_penalty": penalty,
            "adjusted_risk_score": adjusted_risk_score
        }
```

**3. FILE: tests/test_sentiment_risk_bridge.py**
```python
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
```

**4. GIT COMMIT MESSAGE:**
> "feat(jules): implemented SentimentRiskBridge to map market sentiment to risk"