# How-To: Deploy a New Swarm Agent

This guide provides step-by-step instructions for adding a new agent to the System 1 Swarm.

## 1. Create the Agent Class
Create a new file in `core/agents/` (e.g., `core/agents/my_agent.py`). Your agent must inherit from `core.agents.agent_base.AgentBase`.

```python
from core.agents.agent_base import AgentBase
from src.pdil.models import ProvenanceHeader

class MyCustomAgent(AgentBase):
    def __init__(self, config: dict):
        super().__init__(config)

    def execute(self, payload: dict) -> dict:
        # Agent logic here
        return {
            "result": "Success",
            "provenance": ProvenanceHeader(...)
        }
```

## 2. Register with the Orchestrator
Update `core/engine/orchestrator.py` to include your new agent in the Swarm registry.

## 3. Verify Constraints
Ensure your agent outputs valid W3C PROV-O compliance metadata within the `ProvenanceHeader`. Failure to do so will result in the `GovernanceGatekeeper` rejecting the inference.
