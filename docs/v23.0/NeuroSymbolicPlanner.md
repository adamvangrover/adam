# Neuro-Symbolic Planner (v23.0)

## Overview
The Neuro-Symbolic Planner implements the **Plan-on-Graph (PoG)** paradigm. Unlike v22.0 which relied on potentially unstable LLM generation for planning, v23.0 discovers plans by traversing a verifiable **Unified Knowledge Graph (KG)**.

## Components

### 1. Unified Knowledge Graph (`unified_knowledge_graph.py`)
A two-layer graph database:
- **FIBO Layer:** Contains formal financial concepts (e.g., `Company`, `RiskProfile`) and relationships.
- **PROV-O Layer:** Tracks the lineage and provenance of every data point (e.g., `prov_source="SEC EDGAR"`).

Currently implemented using an in-memory `NetworkX` graph for rapid prototyping, simulating a Neo4j backend.

### 2. Planner (`neuro_symbolic_planner.py`)
- **`discover_plan(user_query)`:**
  - Deconstructs the user's intent into a symbolic Start and End node.
  - Finds the shortest verifiable path in the KG.
- **`to_executable_graph(plan)`:**
  - Compiles the symbolic path into a `LangGraph` application.
  - Each node in the path becomes a processing step in the execution graph.

## Usage
```python
from core.v23_graph_engine.neuro_symbolic_planner import NeuroSymbolicPlanner

planner = NeuroSymbolicPlanner()
plan = planner.discover_plan("Analyze Apple Inc. Credit Rating")
app = planner.to_executable_graph(plan)
app.invoke({})
```
