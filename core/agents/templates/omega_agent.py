"""
core/agents/templates/omega_agent.py

Project OMEGA: Base Agent Template.
Enforces the "Adam v25.0 Paradigm" by integrating the Trust Engine (Proof of Thought)
directly into the agent execution lifecycle.

Features:
1. Auto-Hashing: Every input, thought, and output is hashed to the Ledger.
2. Pydantic Schemas: Enforces strict typing for Input/Output.
3. Resilience: Graceful error handling and fallback logic.
"""

import abc
import time
import logging
import uuid
from typing import Dict, Any, Type, Optional
from pydantic import BaseModel, Field

# Pillar 2 Integration
from core.utils.proof_of_thought import ProofOfThoughtLogger
from core.utils.system_logger import SystemLogger

logger = logging.getLogger(__name__)

class AgentInput(BaseModel):
    """Standard input schema for all Omega Agents."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    context: Dict[str, Any] = Field(default_factory=dict)

class AgentOutput(BaseModel):
    """Standard output schema for all Omega Agents."""
    request_id: str
    status: str
    result: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float

class OmegaAgent(abc.ABC):
    """
    Abstract Base Class for Sovereign Agents.
    """
    def __init__(self, name: str, version: str = "v1.0"):
        self.name = name
        self.version = version
        self.pot_logger = ProofOfThoughtLogger()
        self.system_logger = SystemLogger()
        self.logger = logging.getLogger(f"OmegaAgent.{name}")

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """
        The "Sovereign Loop":
        1. Receive & Hash Input (Proof of Request).
        2. Execute Logic (with intermediate "Thoughts").
        3. Validate Output.
        4. Hash Output (Proof of Result).
        """
        start_time = time.time()

        # 1. Proof of Request
        self.pot_logger.log_thought(
            self.name,
            f"INPUT_RECEIVED: {input_data.query}",
            {"request_id": input_data.request_id}
        )

        # System Log: Agent Interaction
        self.system_logger.log_event("AGENT_INTERACTION", {
            "type": "INPUT",
            "agent": self.name,
            "query": input_data.query,
            "request_id": input_data.request_id
        })

        try:
            # 2. Execution (User Implementation)
            self.logger.info(f"Executing logic for {input_data.request_id}...")
            raw_result = await self.process(input_data)

            # 3. Proof of Result
            self.pot_logger.log_thought(
                self.name,
                f"OUTPUT_GENERATED",
                {"request_id": input_data.request_id, "status": "SUCCESS"}
            )

            # System Log: Agent Interaction (Output)
            self.system_logger.log_event("AGENT_INTERACTION", {
                "type": "OUTPUT",
                "agent": self.name,
                "status": "SUCCESS",
                "request_id": input_data.request_id
            })

            return AgentOutput(
                request_id=input_data.request_id,
                status="SUCCESS",
                result=raw_result,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            self.logger.error(f"Execution Failed: {e}", exc_info=True)

            # Proof of Failure
            self.pot_logger.log_thought(
                self.name,
                f"EXECUTION_FAILED: {str(e)}",
                {"request_id": input_data.request_id, "status": "ERROR"}
            )

            # System Log: Agent Interaction (Error)
            self.system_logger.log_event("AGENT_INTERACTION", {
                "type": "ERROR",
                "agent": self.name,
                "error": str(e),
                "request_id": input_data.request_id
            })

            return AgentOutput(
                request_id=input_data.request_id,
                status="ERROR",
                result={"error": str(e)},
                metadata={"trace": "See server logs"},
                execution_time=time.time() - start_time
            )

    @abc.abstractmethod
    async def process(self, input_data: AgentInput) -> Any:
        """
        Implementation specific logic.
        Must be overridden by child agents.
        """
        pass

    def log_thought(self, thought: str, metadata: Dict = None):
        """Helper to log intermediate steps to the blockchain."""
        self.pot_logger.log_thought(self.name, thought, metadata)

# --- Example Implementation ---

class MarketSentimentAgent(OmegaAgent):
    """
    Example Agent: Analyzes market sentiment (Mock).
    """
    async def process(self, input_data: AgentInput) -> Any:
        # Simulate processing time
        self.log_thought("Fetching news from simulated feed...")
        time.sleep(0.1)

        # Simulate analysis
        sentiment_score = 0.75
        self.log_thought(f"Calculated Sentiment: {sentiment_score}", {"model": "FinBERT"})

        return {
            "sentiment": "BULLISH",
            "score": sentiment_score,
            "sources": 5
        }

if __name__ == "__main__":
    import asyncio

    async def test():
        agent = MarketSentimentAgent("SentimentBot_v1")
        inp = AgentInput(query="Analyze $AAPL sentiment")
        result = await agent.execute(inp)
        print(f"Result: {result.model_dump_json(indent=2)}")

    asyncio.run(test())
