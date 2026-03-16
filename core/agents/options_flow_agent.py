import logging
import random
from typing import Any, Dict, Union

from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput


class OptionsFlowAgent(AgentBase):
    """
    Agent responsible for analyzing options flow, specifically unusual volume and put/call ratios.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initializes the OptionsFlowAgent.
        """
        super().__init__(config, **kwargs)
        # Mock mode allows deterministic testing
        self.mock_mode = config.get('mock_mode', False)
        self.mock_put_call_ratio = config.get('mock_put_call_ratio', 1.0)
        self.mock_unusual_volume = config.get('mock_unusual_volume', False)

    async def execute(self, input_data: Union[str, AgentInput, Dict[str, Any]] = None, **kwargs) -> Union[Dict[str, Any], AgentOutput]:
        """
        Executes the options flow analysis.
        Returns a dictionary with sentiment score and details.
        """
        logging.info("OptionsFlowAgent execution started.")

        if self.mock_mode:
            put_call_ratio = self.mock_put_call_ratio
            unusual_volume = self.mock_unusual_volume
        else:
            # Simulate fetching options flow data
            put_call_ratio = round(random.uniform(0.5, 1.5), 2)
            unusual_volume = random.choice([True, False])

        # Calculate sentiment based on put/call ratio
        # A ratio > 1.0 is bearish (more puts than calls), so sentiment should be lower
        # A ratio < 1.0 is bullish (more calls than puts), so sentiment should be higher
        # A simple linear mapping: sentiment = 1.0 - ((put_call_ratio - 0.5) / 1.0)
        # Bounded between 0 and 1
        sentiment_score = 1.0 - ((put_call_ratio - 0.5) / 1.0)
        sentiment_score = max(0.0, min(1.0, sentiment_score))

        # Unusual volume acts as an amplifier to the conviction
        # If unusual volume is true, we amplify the score away from neutral (0.5)
        if unusual_volume:
            if sentiment_score > 0.5:
                sentiment_score = min(1.0, sentiment_score + 0.1) # Boost bullishness
            elif sentiment_score < 0.5:
                sentiment_score = max(0.0, sentiment_score - 0.1) # Boost bearishness

        details = {
            "put_call_ratio": put_call_ratio,
            "unusual_volume": unusual_volume,
            "sentiment_score": sentiment_score
        }

        logging.info(f"Options Flow Analysis - P/C Ratio: {put_call_ratio}, Unusual Vol: {unusual_volume} -> Sentiment: {sentiment_score:.2f}")

        result = {
            "sentiment_score": sentiment_score,
            "details": details,
            "status": "success"
        }

        # Check if we need to return AgentOutput
        is_standard_mode = isinstance(input_data, AgentInput)

        if is_standard_mode:
            return self._format_output(result, input_data.query)

        return result

    def _format_output(self, result: Dict[str, Any], query: str) -> AgentOutput:
        """Helper to format output to AgentOutput."""
        sentiment_score = result.get("sentiment_score", 0.0)
        details = result.get("details", {})

        answer = f"Options Flow Analysis for '{query}':\n"
        answer += f"Sentiment Score: {sentiment_score:.2f} (0.0 Bearish - 1.0 Bullish)\n"
        answer += f"Put/Call Ratio: {details.get('put_call_ratio')}\n"
        answer += f"Unusual Volume Detected: {details.get('unusual_volume')}\n"

        return AgentOutput(
            answer=answer,
            sources=["SimulatedOptionsFlowAPI"],
            confidence=0.8,
            metadata=result
        )
