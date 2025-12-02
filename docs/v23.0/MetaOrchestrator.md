# Meta-Orchestrator (v23.0)

## Overview
The **Meta-Orchestrator** is the "Brain" of the Adam v23.0 system. It acts as the unified entry point for all user queries, intelligent routing them to the most appropriate execution engine based on query complexity.

## Routing Logic

| Complexity | Engine | Use Case |
| :--- | :--- | :--- |
| **LOW** | v21 Sync Tools | "Get stock price of AAPL", "Who is the CEO of MSFT?" |
| **MEDIUM** | v22 Async Message Bus | "Monitor AAPL for news", "Alert me if price drops below $100" |
| **HIGH** | v23 Neuro-Symbolic Planner | "Analyze the credit risk of Apple Inc.", "Plan a diversification strategy" |

## Architecture
- **Planner Integration:** Directly invokes the `NeuroSymbolicPlanner` for high-complexity tasks.
- **Legacy Integration:** Wraps the v22 `HybridOrchestrator` for medium/low complexity tasks.
- **Complexity Assessment:** Currently uses a keyword heuristic; planned upgrade to a BERT-based classifier.

## Usage
```python
from core.engine.meta_orchestrator import MetaOrchestrator

orchestrator = MetaOrchestrator()
result = orchestrator.route_request("Analyze Apple Inc. Credit Risk")
print(result)
```
