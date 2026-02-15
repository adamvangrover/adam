# Tutorial: Building and Using Agents

This tutorial explains how to work with the Autonomous Analyst and other agents in the Adam system.

## Overview

The system uses a neuro-symbolic approach, combining Large Language Models (LLMs) with symbolic reasoning (Knowledge Graphs) to create robust financial agents.

## Key Components

### NeuroSymbolicPlanner
Located in `core/engine/neuro_symbolic_planner.py`.
Responsible for:
- Intent Classification
- Entity Extraction
- Symbolic Plan Discovery
- Natural Language Plan Parsing

### UnifiedKnowledgeGraph
Located in `core/engine/unified_knowledge_graph.py`.
Responsible for:
- Storing relationships (e.g., Company -> Sector -> Macro Indicator).
- Symbolic pathfinding (`find_symbolic_path`).
- Mocking graph ingestion (for testing).

## Creating a New Agent

1.  **Define the Agent Class**: Inherit from a base agent class (if available) or create a standalone class.
2.  **Initialize Components**:
    ```python
    from core.engine.neuro_symbolic_planner import NeuroSymbolicPlanner
    from core.engine.unified_knowledge_graph import UnifiedKnowledgeGraph

    class MyAgent:
        def __init__(self):
            self.planner = NeuroSymbolicPlanner()
            self.kg = UnifiedKnowledgeGraph()
    ```
3.  **Implement Logic**: Use the planner to break down tasks and the KG to reason about entities.

## Example: Analyzing a Ticker

```python
agent = MyAgent()
plan = agent.planner.discover_plan("Analyze AAPL for credit risk")
# Use the plan to execute steps against the KG or external APIs
```

## Testing Agents

Use `pytest` to verify agent behavior. See `tests/test_neuro_symbolic_planner.py` for examples.
