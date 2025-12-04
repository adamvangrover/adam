# Portability & Architecture

Adam v23.5 is architected for **Portability**, **Modularity**, and **Ease of Deployment**. Unlike monolithic legacy systems, Adam is designed as a "Hive Mind" that can be containerized, shipped, and activated in any environment.

## 1. The "Portable Config" Concept

The defining feature of Adam v23.5 is that the agent's "Brain" is decoupled from its "Body" (the codebase).

*   **The Brain (JSON):** The entire cognitive architecture—Role, Directive, Execution Protocol, and Domain Knowledge—is defined in a single JSON file (e.g., `config/Adam_v23.5_Portable_Config.json`).
*   **The Advantage:** This allows users to instantly swap the system's persona.
    *   Need a **Risk Officer**? Load the v22.0 config.
    *   Need a **Growth Investor**? Load the v23.5 config.
    *   This "Hot-Swap" capability means the underlying code doesn't need to change to support radically different use cases.

## 2. The Hybrid Architecture (v21 + v22 + v23)

Adam implements a sophisticated **Hybrid Architecture** that bridges three generations of AI development:

1.  **v21 (Synchronous Legacy):** The original linear orchestration layer, reliable and simple.
2.  **v22 (Asynchronous Message Bus):** A highly scalable, event-driven "Body" capable of handling thousands of messages via RabbitMQ/Redis.
3.  **v23 (Adaptive Graph):** The new "Brain", implemented as a **Cyclical Reasoning Graph** (`LangGraph`). This allows for:
    *   **Iterative Self-Correction:** The agent can critique its own work and loop back to improve it.
    *   **Neuro-Symbolic Planning:** Combining Neural Networks (LLMs) with Symbolic Logic (Knowledge Graphs) for verifiably correct reasoning.

The `MetaOrchestrator` (`core/engine/meta_orchestrator.py`) acts as the traffic controller, routing simple queries to the v21 Legacy layer and complex "Deep Dives" to the v23 Graph Engine.

## 3. The Prime Directive: Bifurcation Strategy

To balance innovation with reliability, the codebase follows a strict **Bifurcation Strategy**:

*   **Path A (Product - `core/vertical_risk_agent`):**
    *   **Focus:** Reliability, Auditability, Business Logic.
    *   **Tech Stack:** Pydantic (Type Safety), Defensive Coding, Comprehensive Logging.
    *   **Use Case:** Production financial analysis where errors are unacceptable.

*   **Path B (Research - `experimental/inference_lab`):**
    *   **Focus:** Velocity, Throughput, Raw Math.
    *   **Tech Stack:** Triton, CUDA, Optimized Kernels.
    *   **Use Case:** High-frequency trading simulations and model optimization research.

## 4. The Universal Ingestor

Data heterogeneity is the enemy of automation. Adam solves this with the **Universal Ingestor** (`core/data_processing/universal_ingestor.py`).

*   **The Problem:** Financial data comes in many shapes: PDF annual reports, CSV market data, JSON API responses, and plain text news articles.
*   **The Solution:** A unified ingestion pipeline that acts as a "Universal Adapter."
*   **How it Works:**
    1.  **Scanning:** The ingestor scans target directories or feeds.
    2.  **Normalization:** It parses diverse file types (`.pdf`, `.csv`, `.json`, `.md`) using specialized handlers.
    3.  **Standardization:** All data is converted into a **Standardized JSONL Format** (The "Gold Standard").
    4.  **Ingestion:** The system consumes this clean, uniform stream, making the core reasoning logic **Data-Agnostic**.

## 5. Containerization (Docker)

Adam is "Cloud-Native" by design. The repository includes full containerization support.

*   **`Dockerfile`:** Defines the runtime environment, ensuring Python 3.10+, system dependencies, and exact package versions are present.
*   **`docker-compose.yml`:** Orchestrates the multi-container setup, spinning up:
    *   **The Core Agent:** The Python backend running the Meta Orchestrator.
    *   **Redis:** For the high-speed Knowledge Graph cache.
    *   **Neo4j (Optional):** For persistent graph storage.

This ensures that Adam runs identically on a developer's laptop, an on-premise server, or a cloud Kubernetes cluster.
