# Bolt's Journal

## 2024-05-22 - [Double Graph Identity]
**Learning:** The codebase contains two different `UnifiedKnowledgeGraph` classes in different directories (`core/engine` vs `core/v23_graph_engine`). `NeuroSymbolicPlanner` uses the one in `core/engine`. This duplication is a trap.
**Action:** Always verify imports to confirm which file is actually being used before optimizing.

## 2024-05-22 - [Singleton Graph Loading]
**Learning:** `UnifiedKnowledgeGraph` re-parses JSON and rebuilds the graph on every instantiation. Since `NeuroSymbolicPlanner` instantiates it in `__init__` and is often transient (or at least could be), this is a major bottleneck.
**Action:** Implement a module-level cache for the graph structure to avoid redundant I/O and graph construction.

## 2024-05-24 - [Unused Components & Testing]
**Learning:** Found `KnowledgeGraphVisualizer` was unused and untestable via the app. `react-force-graph-2d` requires mocking in JSDOM.
**Action:** Always check if a component is mounted before trying to verify it visually. Use unit tests with mocks for library-heavy components.

## 2025-05-18 - React Re-render Patterns
**Learning:** In interactive components like Terminals, keeping the large list state (history) in the same component as the high-frequency input state (typing) causes massive re-renders.
**Action:** Always separate "Display State" (slow changing) from "Input State" (fast changing) into separate components, or use `React.memo` on the Display component.

## 2025-05-22 - Debouncing High-Frequency Simulation Inputs
**Learning:** The `ScenarioSimulator` component was triggering a "Monte Carlo" simulation on every `input` change event from a range slider. In React, range inputs fire `onChange` continuously during a drag. This causes massive calculation overhead (O(N) * updates) for what should be a terminal state calculation.
**Action:** Always debounce inputs that trigger heavy calculations, especially range sliders or text search fields. Used a standard `useEffect` debounce pattern with a `useRef` mount check to decouple UI state (instant) from simulation state (delayed).

## 2024-05-21 - [Costly Database Connections in Request Handlers]
**Learning:** Initializing database drivers (like Neo4j) inside request handlers creates a new connection pool for every request, leading to severe performance degradation under load.
**Action:** Always verify database driver initialization scope. Move to singleton or app-level context.

## 2025-05-23 - [NetworkX Batching]
**Learning:** `graph.add_nodes_from` with attributes is significantly faster (~30%) than looping `graph.add_node` for large datasets (100k+ nodes), likely due to reduced function call overhead and internal optimizations.
**Action:** When ingesting large subgraphs, always batch node/edge creation into lists.
