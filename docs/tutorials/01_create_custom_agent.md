# Tutorial: Creating a Custom Agent in Adam v26.0

This tutorial guides you through creating a new, specialized agent that integrates with the Adam v26.0 ecosystem.

## Prerequisites
*   Understanding of Python 3.10+ and Pydantic.
*   Familiarity with the `core/agents/` directory structure.

## Step 1: Define the Purpose
Let's create a **`CryptoSentimentAgent`**.
*   **Input:** A cryptocurrency symbol (e.g., "BTC").
*   **Task:** Fetch recent news/tweets and calculate a sentiment score.
*   **Output:** A structured `SentimentReport` object.

## Step 2: Define the Schema
In `core/schemas/agent_schema.py` (or a new file), define your input/output if the standard ones don't suffice. For now, we'll use standard inputs.

## Step 3: Implement the Agent Class
Create `core/agents/specialized/crypto_sentiment_agent.py`.

```python
import logging
from typing import Dict, Any
from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput

logger = logging.getLogger(__name__)

class CryptoSentimentAgent(AgentBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.persona = "Crypto Market Analyst"
        # Load specific configs, API keys, etc.

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        logger.info(f"Analyzing sentiment for {input_data.query}")

        # 1. Fetch Data (Mocking external call)
        # In real life, use self.tools.call("fetch_crypto_news", ...)
        raw_news = ["Bitcoin hits new high", "Regulatory concerns loom"]

        # 2. Process (Sentiment Analysis)
        score = 0.8  # Mock calculation

        # 3. Construct Output
        return AgentOutput(
            answer=f"Sentiment for {input_data.query} is Bullish.",
            confidence=0.85,
            sources=["Twitter", "CoinDesk"],
            metadata={
                "raw_score": score,
                "news_count": len(raw_news)
            }
        )
```

## Step 4: Register the Agent
Update `core/system/agent_orchestrator.py` to include your new class in the loader.

```python
from core.agents.specialized.crypto_sentiment_agent import CryptoSentimentAgent
# ...
self.agent_registry["crypto_sentiment"] = CryptoSentimentAgent
```

## Step 5: Test the Agent
Create a test script in `tests/test_new_agent.py`.

```python
import pytest
from core.agents.specialized.crypto_sentiment_agent import CryptoSentimentAgent
from core.schemas.agent_schema import AgentInput

@pytest.mark.asyncio
async def test_crypto_agent():
    agent = CryptoSentimentAgent(config={})
    input_data = AgentInput(query="BTC")
    result = await agent.execute(input_data)

    assert result.confidence > 0.0
    assert "Bullish" in result.answer
```

## Step 6: Integration
You can now use this agent in a Workflow Graph or call it via the Swarm.
