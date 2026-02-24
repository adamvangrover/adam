# Adam v26.0: The Three-Layer Architecture

Adam v26.0 is designed as a **Neuro-Symbolic Sovereign**, composed of three distinct, decoupled layers. This architecture allows each component to operate independently, scale horizontally, and be swapped out without affecting the others.

## 1. Intelligence Layer (System 2)
*   **Role:** Reasoning, Planning, and Decision Making.
*   **Core Component:** `core.agents.risk_assessment_agent`, `core.engine.neuro_symbolic_planner`
*   **Function:** Accepts structured data, applies business logic (via `jsonLogic` or Python), and outputs decisions with provenance.
*   **Example:** [examples/core_functionality/01_intelligence_layer.py](../../examples/core_functionality/01_intelligence_layer.py)

## 2. Compute Layer (System 3)
*   **Role:** Simulation, World Modeling, and Heavy Calculation.
*   **Core Component:** `core.engine.live_mock_engine`, `core.math.probability_models`
*   **Function:** Runs Monte Carlo simulations, calculates VaR, generates credit memos, and provides market pulses.
*   **Environment Rotation:** Can switch between `SIMULATION` (System 3) and `LIVE` (System 1) execution backends via `EngineFactory`.
*   **Example:** [examples/core_functionality/02_compute_layer.py](../../examples/core_functionality/02_compute_layer.py)

## 3. Data Layer (System 1)
*   **Role:** Ingestion, Perception, and ETL.
*   **Core Component:** `core.ingestion.semantic_chunker`, `core.knowledge_graph`
*   **Function:** Ingests raw unstructured data (PDFs, News, Feeds), chunks it semantically, and stores it in the Knowledge Graph or Vector Store.
*   **Example:** [examples/core_functionality/03_data_layer.py](../../examples/core_functionality/03_data_layer.py)

---

## Inter-Layer Communication

While decoupled, the layers communicate via standard JSON contracts:

1.  **Data Layer** produces **Artifacts** (Cleaned JSON/Text).
2.  **Compute Layer** consumes Artifacts to produce **Metrics** (Risk Scores, Valuations).
3.  **Intelligence Layer** consumes Metrics to produce **Decisions** (Buy/Sell, Approve/Reject).

## Provenance & Logging

All layers utilize the `ProvenanceLogger` to ensure every action is traceable.
*   **Data:** Logs source and hash of raw input.
*   **Compute:** Logs model version and parameters.
*   **Intelligence:** Logs reasoning chain and final decision.
