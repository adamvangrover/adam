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
*   **Role:** Ingestion, Multimodal Perception, and ETL.
*   **Core Component:** `core.ingestion.semantic_chunker`, `core.knowledge_graph`
*   **Function:** Ingests raw unstructured and multimodal data (PDFs, News, Feeds, Images, Audio Transcripts), chunks it semantically, and stores it in the Knowledge Graph or Vector Store.
*   **Example:** [examples/core_functionality/03_data_layer.py](../../examples/core_functionality/03_data_layer.py)

## 4. Integration Layer
*   **Role:** Governance, Validation, and State Gatekeeping.
*   **Core Component:** `core.governance.pdil_gatekeeper`
*   **Function:** The **Probabilistic-to-Deterministic Integration Layer (PDIL)** enforces domain boundaries, verifying that any probabilistic outputs from Neural Swarms strictly map to deterministic inputs expected by the Compute or Intelligence Layers.

---

## Inter-Layer Communication

While decoupled, the layers communicate via standard JSON contracts routed through the PDIL:

1.  **Data Layer** produces **Artifacts** (Cleaned JSON/Text/Extracted Multimodal Features).
2.  **Integration Layer (PDIL)** validates and transforms Artifacts into structured **Deterministic Parameters**.
3.  **Compute Layer** consumes Parameters to produce **Metrics** (Risk Scores, Valuations).
4.  **Intelligence Layer** consumes Metrics to produce **Decisions** (Buy/Sell, Approve/Reject).

## Provenance & Logging

All layers utilize the `ProvenanceLogger` to ensure every action is traceable.
*   **Data:** Logs source and hash of raw input.
*   **Compute:** Logs model version and parameters.
*   **Intelligence:** Logs reasoning chain and final decision.
