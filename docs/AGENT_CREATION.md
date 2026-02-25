# Agent Creation Guide

## Overview
Agents in Adam are specialized units of intelligence. They can be simple (single-prompt) or complex (multi-step reasoning).

## Step-by-Step Guide

### 1. Define the Agent Interface
Create a new class in `core/agents/specialized/` inheriting from `AgentBase`.

```python
from core.agents.agent_base import AgentBase

class CryptoAnalystAgent(AgentBase):
    def __init__(self):
        super().__init__(name="CryptoAnalyst", role="Cryptocurrency Market Specialist")
```

### 2. Implement Capabilities
Define the tools the agent can use.

```python
    def get_capabilities(self):
        return [
            "analyze_token_metrics",
            "check_onchain_volume"
        ]
```

### 3. Implement Reasoning Logic
Override the `process` method.

```python
    def process(self, query: str, context: dict):
        # 1. Fetch Data
        metrics = self.tools.fetch_metrics(query)

        # 2. Reason (LLM Call)
        analysis = self.llm.generate(
            prompt=f"Analyze {query} given metrics: {metrics}",
            system_prompt="You are a skeptical crypto analyst."
        )

        return analysis
```

### 4. Register the Agent
Add the agent to `core/agents/registry.py` (or equivalent configuration) so the Orchestrator can discover it.

### 5. Test
Write a unit test in `tests/agents/test_crypto_agent.py` to verify behavior with mock data.
