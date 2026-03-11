import logging
from typing import Any, Dict, Union
from pydantic import BaseModel, Field
import random

from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput

logger = logging.getLogger(__name__)

class InsiderActivityData(BaseModel):
    buy_volume: float = Field(default=0.0, description="Total dollar volume of insider buys")
    sell_volume: float = Field(default=0.0, description="Total dollar volume of insider sells")
    cluster_buys: bool = Field(default=False, description="True if multiple insiders are buying within a short window")
    officer_buys: int = Field(default=0, description="Number of C-level officers buying")

class InsiderActivityAgent(AgentBase):
    """
    Agent responsible for monitoring corporate insider activity (Form 4 filings).
    It tracks buy/sell ratios and cluster buying behavior to gauge internal conviction.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initializes the InsiderActivityAgent.
        """
        super().__init__(config, **kwargs)
        self.mock_mode = config.get('mock_mode', False)

    async def execute(self, input_data: Union[str, AgentInput, Dict[str, Any]] = None, **kwargs) -> Union[Dict[str, Any], AgentOutput]:
        """
        Executes the insider activity analysis.
        Calculates sentiment based on buy/sell ratios and cluster buying signals.
        """
        logger.info("InsiderActivityAgent execution started.")
        query = ""
        is_standard_mode = False

        if input_data is not None:
            if isinstance(input_data, AgentInput):
                query = input_data.query
                is_standard_mode = True
            elif isinstance(input_data, str):
                query = input_data
            elif isinstance(input_data, dict):
                query = input_data.get("query", "")

        # In a real system, we would fetch Form 4 data here for the queried ticker/market.
        data = self._fetch_insider_data(query)
        
        # 1. Base Sentiment from Buy/Sell Ratio
        total_volume = data.buy_volume + data.sell_volume
        if total_volume == 0:
            sentiment_score = 0.5  # Neutral if no activity
        else:
            # Map buy ratio to 0.0 - 1.0 (0.5 is neutral)
            buy_ratio = data.buy_volume / total_volume
            # Normalize: let's say a buy ratio of > 0.2 is actually bullish since insiders typically sell more for liquidity
            if buy_ratio > 0.3:
                sentiment_score = 0.5 + ((buy_ratio - 0.3) / 0.7) * 0.5
            else:
                sentiment_score = (buy_ratio / 0.3) * 0.5
                
        # 2. Adjust for Cluster Buys & Officer Buys
        if data.cluster_buys:
            sentiment_score += 0.15
        
        if data.officer_buys > 1:
            sentiment_score += 0.1
            
        # 3. Cap the score
        sentiment_score = max(0.0, min(1.0, sentiment_score))

        details = {
            "buy_volume": data.buy_volume,
            "sell_volume": data.sell_volume,
            "cluster_buys": data.cluster_buys,
            "officer_buys": data.officer_buys,
            "sentiment_score": sentiment_score
        }

        logger.info(f"Insider Activity Analysis -> Sentiment: {sentiment_score:.2f}")

        result = {
            "sentiment_score": sentiment_score,
            "details": details,
            "status": "success"
        }

        if is_standard_mode:
            return self._format_output(result, query)

        return result

    def _fetch_insider_data(self, query: str) -> InsiderActivityData:
        """Simulates fetching SEC Form 4 insider data."""
        if self.mock_mode:
             return InsiderActivityData(
                 buy_volume=1500000.0,
                 sell_volume=500000.0,
                 cluster_buys=True,
                 officer_buys=2
             )
        # Add some randomness for non-mock but simulated execution
        return InsiderActivityData(
             buy_volume=random.uniform(0, 5000000),
             sell_volume=random.uniform(100000, 10000000),
             cluster_buys=random.choice([True, False]),
             officer_buys=random.randint(0, 3)
        )

    def _format_output(self, result: Dict[str, Any], query: str) -> AgentOutput:
        """Helper to format output to AgentOutput."""
        sentiment_score = result.get("sentiment_score", 0.0)
        details = result.get("details", {})

        answer = f"Insider Activity Analysis for '{query}':\n"
        answer += f"Sentiment Score: {sentiment_score:.2f} (0.0 Bearish - 1.0 Bullish)\n"
        answer += f"Buy Volume: ${details.get('buy_volume'):,.2f}\n"
        answer += f"Sell Volume: ${details.get('sell_volume'):,.2f}\n"
        answer += f"Cluster Buys Detected: {details.get('cluster_buys')}\n"

        return AgentOutput(
            answer=answer,
            sources=["SimulatedSECForm4API"],
            confidence=0.85,
            metadata=result
        )
