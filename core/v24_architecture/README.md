# Adam v24.0 "Production-Grade" Architecture

This directory contains the reference implementation for the **Adam v24.0 Architecture**, designed to transition the system from a "Showcase" prototype (v23.5) to a robust, autonomous financial platform.

## Overview

The architecture is built on four pillars, as detailed in `docs/v24_remediation_plan.md`:

1.  **The Brain (Cognitive Core):**
    *   `brain/semantic_router.py`: Replaces keyword matching with Vector Embeddings (MiniLM) for intent classification.
    *   `brain/rag_planner.py`: Implements RAG-Guided Planning with NER and Vector Anchoring.

2.  **The Reasoning Engine:**
    *   `reasoning/robust_graph.py`: A stateful `LangGraph` workflow with persistence and loop limits.
    *   `reasoning/self_reflection.py`: A "Senior Editor" agent that critiques drafts against a "Constitution".

3.  **Data Integrity:**
    *   `integrity/schema.py`: Strict Pydantic models for verifying agent outputs.
    *   `integrity/conviction.py`: Semantic Conviction Scoring using Cross-Encoders.

4.  **Infrastructure:**
    *   `infrastructure/message_bus.py`: Abstract interface for async messaging (RabbitMQ/Redis).
    *   `infrastructure/service_definitions.py`: Configuration for the microservices topology.

## Dependencies

To run this module, the following additional packages are required:

```text
langgraph
sentence-transformers
pydantic>=2.0
numpy
fastapi
uvicorn
psycopg2-binary  # For Postgres persistence
```

## Usage Example

```python
import asyncio
from core.v24_architecture.brain.semantic_router import SemanticRouter
from core.v24_architecture.reasoning.robust_graph import build_robust_graph

async def main():
    # 1. Route Intent
    router = SemanticRouter()
    intent = router.route("Analyze the liquidity risk of TSLA")
    print(f"Routed to: {intent}")

    # 2. Execute Reasoning Graph
    graph = build_robust_graph()
    initial_state = {
        "request": "Analyze TSLA liquidity",
        "iteration_count": 0,
        "is_complete": False
    }

    # Run the graph (async)
    async for event in graph.astream(initial_state):
        print(event)

if __name__ == "__main__":
    asyncio.run(main())
```

## Migration Status

This module is **ADDITIVE**. It does not replace the existing `v23_graph_engine` or `agents` in the root `core/` directory. It is intended to run in parallel or as a replacement target for future refactoring.
