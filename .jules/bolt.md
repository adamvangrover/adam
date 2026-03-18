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

## 2025-12-25 - [Optimized Loading Component Style Injection]
**Learning:** React components that inject <style> tags on every render cause unnecessary style recalculations and layout thrashing. Moving static styles to global CSS files (e.g., index.css) is a simple but effective optimization.
**Action:** Always check for <style> tags inside functional components during code review.

## 2025-12-28 - [Duplicate Knowledge Graph Identity]
**Learning:** A critical "trap" was identified where `core/engine/unified_knowledge_graph.py` and `core/v23_graph_engine/unified_knowledge_graph.py` contain identical but distinct classes. Optimizing one without the other leads to inconsistent system behavior depending on which module imports it.
**Action:** When working on core components, always `grep` for potential duplicates or moved files to ensure the optimization is applied globally.

## 2026-01-08 - [Regex Compilation in Loops]
**Learning:** `re.search` and `re.findall` inside loops or frequent calls compile the regex pattern implicitly, even if cached. Explicitly pre-compiling patterns to class constants avoids this overhead and also prevents unnecessary string operations like `desc.lower()` when `re.IGNORECASE` can be used.
**Action:** Always pre-compile regex patterns as module or class constants.

## 2026-02-18 - [Duplicate Scrubber Logic]
**Learning:** Found `core/data_processing/utils.py` duplicated logic from `core/data_processing/universal_ingestor.py` but missed critical optimizations (like pre-compiled regexes).
**Action:** When optimizing, check for duplicate utility classes that might have diverged.

## 2026-02-01 - [React List Reconciliation]
**Learning:** `AgentIntercom` used array index as key for a sliding window list (prepending items). This forces React to re-render *every* item on every update because the content at index 0 changes.
**Action:** Always use stable unique IDs (UUIDs) for list items, especially when the list order changes or items are prepended/appended. Modified backend to supply IDs.

## 2026-02-27 - [NumPy Vectorization for Monte Carlo]
**Learning:** Pure Python loops for Monte Carlo simulations (Geometric Brownian Motion) are extremely slow (~2.9s for 5k simulations). Vectorizing with `np.random.normal`, `np.cumsum` (in-place), and `np.exp` reduced this to ~0.36s (8x speedup).
**Action:** Always vectorize numerical simulations involving large iterations using NumPy array operations.

## 2026-03-05 - [Yahoo Finance Batching]
**Learning:** `yf.Ticker(symbol).history()` in a loop is purely sequential and network-bound. `yf.download(tickers, ...)` uses threads internally and is ~4.5x faster (0.33s vs 1.49s for 5 tickers).
**Action:** Always use `yf.download` with a list of tickers for multi-asset analysis, but handle the resulting MultiIndex DataFrame carefully (top level is Ticker if `group_by='ticker'`).

## 2026-10-25 - [Vectorized Quantum Monte Carlo]
**Learning:** The `QuantumMonteCarloEngine` in `core/risk_engine/quantum_monte_carlo.py` was using a Python loop to generate simulations, which is significantly slower than using NumPy's vectorized operations.
**Action:** Replaced the loop with a vectorized `run_circuit_batch` method in `SimulatorBackend`, resulting in a ~4.6x speedup (0.35s -> 0.076s for 100k simulations). Always check for loops in numerical simulations and vectorize where possible.

## 2026-10-26 - [DOM Traversal Optimization]
**Learning:** `TreeWalker` in `showcase/js/dashboard-logic.js` was traversing the entire document and running regex matching on every text node, even though only the first 20 matches were used. This caused unnecessary processing on large pages (~58ms vs ~0.3ms for 5000 nodes).
**Action:** Always implement an early exit condition when scanning the DOM for a limited number of matches.

## 2024-05-19 - React Memoization on Global Store Subscriptions
**Learning:** In dashboards subscribing to global store states (like Zustand), changes to root metrics (e.g., `networkLoad`) cause the entire component to re-render, including mapping over lists. Wrapping list child items (like `AgentCell`) in `React.memo` effectively isolates them, ensuring they only re-render if their direct prop ref changes.
**Action:** Always verify if a parent component maps over items and subscribes to unrelated state. Use `React.memo` on the list items to avoid O(N) re-renders.

## 2025-05-28 - [Vectorized pandas DataFrame to dict conversion]
**Learning:** `df.iterrows()` inside pandas is notoriously slow because it converts each row to a Series. The iteration inside `YFinanceMarketData` methods introduced a massive overhead. Refactoring the iteration loop into a vectorized approach (`df.rename`, modifying `df.index`, followed by `df.reset_index(names="date")[cols].to_dict(orient="records")`) achieves roughly a 4-12x performance boost with no functional changes.
**Action:** Always prefer `to_dict(orient="records")` for DataFrame iterations that construct output object lists, particularly when fetching and returning large blocks of market data APIs.

## 2025-06-12 - [Vectorized pandas DataFrame logic in WhaleScanner]
**Learning:** `merged.iterrows()` inside `WhaleScanner.calculate_fund_sentiment` is notoriously slow because it converts each row to a Series. Refactoring the iteration loop into a vectorized approach (`combine_first`, `fillna`, mapping columns, and `to_dict('records')`) achieves a roughly 5x performance boost with no functional changes.
**Action:** Always prefer `to_dict(orient="records")` on filtered or mapped DataFrame subsets over `.iterrows()` when generating parsed domain models from external tabular data.

## 2025-06-13 - [Vectorized historical data fetching in DataFetcher]
**Learning:** `history_reset.iterrows()` inside `DataFetcher.fetch_historical_data` was a significant bottleneck when fetching years of daily data, taking ~0.38 seconds for 1 year of AAPL data. Refactoring to a vectorized approach mapping columns and formatting datetime natively, then using `to_dict(orient="records")`, reduced this to ~0.04 seconds (~10x speedup).
**Action:** Consistently avoid `df.iterrows()` in data pipeline and ingestion methods. Use vectorized pandas operations and `to_dict(orient="records")` to build dictionaries.

## 2025-03-10 - Vectorized DataFrame filtering by Multi-Index
**Learning:** `df.iterrows()` when iterating to check if tuple pairs exist in a `set` is extremely slow. In `InstitutionalRadarAnalytics.detect_cluster_buys`, using a `for _, row in curr_df.iterrows():` loop checking `(row["fund_name"], row["cusip"]) not in prev_holdings` was a major bottleneck.
**Action:** Always prefer `mask = ~curr_df.set_index(["col1", "col2"]).index.isin(set_of_tuples)` instead of `iterrows()` for multi-column exclusion/inclusion filtering against a set of tuples.
## 2024-05-14 - Replace pandas iterrows with to_dict('index') for Nested MultiIndex Data
**Learning:** Using `iterrows()` combined with repeated MultiIndex `.loc[]` lookups in a nested loop is a major pandas performance anti-pattern. When you need to transform a `pandas.DataFrame` into a nested dictionary, breaking it down column-by-column (or ticker-by-ticker) with `dropna()` and using `to_dict('index')` is ~10-100x faster than iterating row-by-row. Also, remember to preserve the initialization logic (e.g. empty date dicts) and handle potential NaN casting values (like `int(NaN)` for volume).
**Action:** Always avoid `iterrows()` for data transformation. Instead, apply vectorized column filtering and leverage pandas' optimized `to_dict()` export functions to serialize data.

## 2025-06-14 - [Vectorized pandas DataFrame logic in Agent Improvement and Ingestion]
**Learning:** `df.iterrows()` was used in `AgentImprovementPipeline` and `InstitutionalRadarAnalytics` causing significant performance bottlenecks due to row-by-row Series conversion. Replacing `df.iterrows()` with `zip(df.index, df.to_dict(orient="records"))` and `df.to_dict(orient="records")` yielded significant speedups (~10x) with zero logic changes.
**Action:** Always prefer `to_dict(orient="records")` on filtered or mapped DataFrames over `.iterrows()` loops.
