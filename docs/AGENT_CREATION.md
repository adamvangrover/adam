# Agent Creation Guide (v30.1)

## Overview
Agents in ADAM (Autonomous Deterministic Alpha Matrix) are specialized units of intelligence that bridge stochastic language processing (System 1) with deterministic execution (System 2).

To prevent hallucinations and guarantee auditability, **all agents must adhere to the Probabilistic-to-Deterministic Integration Layer (PDIL)** constraints.

## Step-by-Step Guide

### 1. Define the Agent Class
Create a new Python file in `core/agents/` (e.g., `yield_farming_agent.py`). Inherit from `AgentBase`. The constructor must accept a single `config` dictionary.

```python
import logging
import datetime
from typing import Any, Dict, Optional
from core.schemas.agent_schema import AgentInput, AgentOutput
from core.agents.agent_base import AgentBase
from src.pdil.models import ProvenanceHeader

logger = logging.getLogger(__name__)

class YieldFarmingAgent(AgentBase):
    def __init__(self, config: Optional[Dict[str, Any]] = None, kernel: Optional[Any] = None):
        if config is None:
            config = {"name": "YieldFarmingAgent"}
        super().__init__(config, kernel=kernel)
```

### 2. Implement the Execute Method
Override the `execute` method. It must accept an `AgentInput` schema and return an `AgentOutput` schema.

```python
    async def execute(self, input_data: AgentInput) -> AgentOutput:
        query = input_data.query.lower()

        # 1. Agent Logic (e.g., LLM call, API fetch)
        answer = "Recommended allocation: Curve 3pool"
        confidence = 0.85
```

### 3. Enforce Provenance (Mandatory)
To satisfy W3C PROV-O compliance and the PDIL, every `AgentOutput` **must** include a `provenance_trace` using the `ProvenanceHeader` schema.

```python
        # 2. Generate Provenance
        provenance = ProvenanceHeader(
            git_commit_hash="current_commit_hash",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            content_hash="hash_of_the_content",
            jsonLogic_version="1.0.0",
            confidence_score=confidence,
            derivation_path="yield_farming_agent",
            source_data_object="Live_Telemetry_Engine_YieldFarming"
        )

        # 3. Return Strictly Typed Output
        return AgentOutput(
            answer=answer,
            confidence=confidence,
            metadata={"status": "success"},
            provenance_trace=provenance
        )
```

### 4. Registration and Testing
*   **Evaluation-First Development**: Before adding to the swarm, the agent must pass regression tests.
*   **Testing**: Write a `pytest` file in `tests/unit/`. Use the `check_grounding` helper to verify that outputs contain valid `ProvenanceHeader` references to their source data objects.
