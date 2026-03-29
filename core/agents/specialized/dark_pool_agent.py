from typing import Any, Dict, List, Optional
import logging
from pydantic import Field

from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput

logger = logging.getLogger(__name__)

class DarkPoolAgentInput(AgentInput):
    """
    Input schema for the DarkPoolAgent. Extends AgentInput.
    """
    total_volume: float = Field(..., description="Total market volume for the ticker.")
    dark_pool_volume: float = Field(..., description="Volume executed off-exchange (dark pools).")
    average_dark_pool_ratio: float = Field(0.4, description="Historical average dark pool ratio (e.g., 40%).")

class DarkPoolAgentOutput(AgentOutput):
    """
    Output schema for the DarkPoolAgent. Extends AgentOutput.
    """
    dark_pool_ratio: float = Field(..., description="Current dark pool volume to total volume ratio.")
    anomaly_score: float = Field(..., description="Severity of the dark pool volume deviation.")
    is_anomaly: bool = Field(..., description="True if the anomaly score exceeds threshold.")

class DarkPoolAgent(AgentBase):
    """
    DarkPoolAgent analyzes off-exchange (dark pool) trading volume
    to detect hidden institutional accumulation or distribution.
    It integrates with the system by providing an anomaly score based on volume inputs.
    """
    def __init__(self, config: Dict[str, Any], constitution: Optional[Dict[str, Any]] = None, kernel: Optional[Any] = None):
        super().__init__(config, constitution=constitution, kernel=kernel)
        self.anomaly_threshold = self.config.get("anomaly_threshold", 0.15) # 15% deviation

    async def execute(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes the DarkPoolAgent logic.

        Args:
            Accepts DarkPoolAgentInput parameters via kwargs.

        Returns:
            Dict mapping to DarkPoolAgentOutput.
        """
        try:
            # Provide default query and context if not provided
            if "query" not in kwargs:
                kwargs["query"] = "Analyze dark pool volume"
            if "context" not in kwargs:
                kwargs["context"] = {}

            # Validate Input using Pydantic model
            input_data = DarkPoolAgentInput(**kwargs)

            # Prevent division by zero
            if input_data.total_volume <= 0:
                raise ValueError("Total volume must be greater than zero.")

            # Calculate metrics
            current_ratio = input_data.dark_pool_volume / input_data.total_volume
            deviation = current_ratio - input_data.average_dark_pool_ratio

            # Determine anomaly score and flag
            # Positive deviation means more dark pool activity than usual
            anomaly_score = deviation
            is_anomaly = abs(deviation) >= self.anomaly_threshold

            # Format the qualitative answer
            if is_anomaly and deviation > 0:
                answer = f"High dark pool activity detected. Ratio is {current_ratio:.2%}, which is {deviation:.2%} above the historical average of {input_data.average_dark_pool_ratio:.2%}."
            elif is_anomaly and deviation < 0:
                answer = f"Unusually low dark pool activity detected. Ratio is {current_ratio:.2%}, which is {abs(deviation):.2%} below the historical average of {input_data.average_dark_pool_ratio:.2%}."
            else:
                answer = f"Dark pool activity is within normal ranges. Ratio is {current_ratio:.2%}."

            # Calculate Conviction (Confidence) based on the magnitude of the deviation
            # Max conviction at 3x threshold
            confidence = min(1.0, abs(deviation) / (self.anomaly_threshold * 3.0) + 0.5)

            # Construct Output using Pydantic model
            output_data = DarkPoolAgentOutput(
                answer=answer,
                sources=["Dark Pool Volume Feed"],
                confidence=confidence,
                metadata={"deviation": deviation},
                dark_pool_ratio=current_ratio,
                anomaly_score=anomaly_score,
                is_anomaly=is_anomaly
            )

            # Publish insight to swarm memory if it's an anomaly with high confidence
            if output_data.is_anomaly and output_data.confidence > 0.8:
                self.publish_insight("DarkPoolActivity", output_data.answer, output_data.confidence)

            return output_data.model_dump()

        except Exception as e:
            logger.error(f"DarkPoolAgent execution failed: {e}")
            raise
