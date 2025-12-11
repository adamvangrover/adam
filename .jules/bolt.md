# Bolt's Journal

## 2024-05-22 - [Double Graph Identity]
**Learning:** The codebase contains two different `UnifiedKnowledgeGraph` classes in different directories (`core/engine` vs `core/v23_graph_engine`). `NeuroSymbolicPlanner` uses the one in `core/engine`. This duplication is a trap.
**Action:** Always verify imports to confirm which file is actually being used before optimizing.

## 2024-05-22 - [Singleton Graph Loading]
**Learning:** `UnifiedKnowledgeGraph` re-parses JSON and rebuilds the graph on every instantiation. Since `NeuroSymbolicPlanner` instantiates it in `__init__` and is often transient (or at least could be), this is a major bottleneck.
**Action:** Implement a module-level cache for the graph structure to avoid redundant I/O and graph construction.
