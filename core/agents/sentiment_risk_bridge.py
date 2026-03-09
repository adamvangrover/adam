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
