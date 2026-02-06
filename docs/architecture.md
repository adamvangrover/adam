# Adam v26.0 Architecture: The Neuro-Symbolic Sovereign

## 1. Executive Summary
Adam v26.0 represents a paradigm shift from **Generative AI** to **Agentic AI**. It is designed not just to chat, but to *execute* complex financial workflows with fiduciary-grade reliability.

The core architectural innovation is the **Hybrid Cognitive Engine**:
*   **System 1 (The Swarm):** Fast, asynchronous, massively parallel. Handles news ingestion, sentiment scoring, and data fetching.
*   **System 2 (The Graph):** Slow, synchronous, stateful. Handles deep reasoning, multi-step planning, and risk adjudication.

## 2. High-Level Diagram

```mermaid
graph TD
    User[User / API] --> Meta[Meta Orchestrator]

    subgraph "System 1: The Swarm (Async)"
        Meta -.->|Fast Query| Swarm[Swarm Manager]
        Swarm --> Worker1[News Bot]
        Swarm --> Worker2[Data Fetcher]
        Swarm --> Worker3[Sentiment Scorer]
        Worker1 & Worker2 & Worker3 --> MessageBus[Redis / Memory]
    end

    subgraph "System 2: The Graph (Reasoning)"
        Meta ==>|Deep Dive| Planner[Neuro-Symbolic Planner]
        Planner --> Graph[Execution Graph (DAG)]

        Graph --> NodeA[Fundamental Analysis]
        Graph --> NodeB[Risk Modeling]
        Graph --> NodeC[Legal Review]

        NodeA & NodeB & NodeC --> Consensus[Consensus Engine]
        Consensus -->|Low Conviction| Planner
        Consensus -->|High Conviction| FinalOutput
    end

    MessageBus -.-> Graph
```

## 3. Core Subsystems

### 3.1 The Neuro-Symbolic Planner (`core/engine`)
Unlike standard "Auto-GPT" loops which can get stuck, Adam uses a **Planner-Executor-Critic** pattern.
1.  **Plan:** The user's goal is broken into a fixed DAG (Directed Acyclic Graph).
2.  **Execute:** Agents run the nodes.
3.  **Critique:** A separate LLM call validates the output against the plan.

### 3.2 The Credit Sentinel (`core/credit_sentinel`)
This is the "Distressed Debt" specialist module.
*   **Input:** 10-K/10-Q filings, Earnings Call Transcripts.
*   **Process:**
    *   **Ratio Calculator:** Deterministic Python functions (System 2) calculate EBITDA, Leverage, etc.
    *   **Trend Watcher:** Pandas-based time-series analysis.
    *   **Risk Analyst:** LLM-based qualitative assessment of Management Discussion & Analysis (MD&A).
*   **Output:** A regulatory-grade Investment Memo.

### 3.3 The Universal Ingestor (`core/data_processing`)
Data integrity is paramount.
*   **Pipeline:** Raw File -> Text Extraction (Docling) -> PII Redaction -> Chunking -> Vectorization.
*   **Semantic Conviction:** Every chunk is scored for reliability.

### 3.4 The Neural Dashboard (`webapp/`)
A decoupled frontend that provides visibility into the "Black Box".
*   **Visualizing Thought:** The dashboard subscribes to the `AgentIntercom` channel to show real-time reasoning steps.

## 4. Tooling & Model Context Protocol (MCP)

Adam v26.0 adopts the **Model Context Protocol (MCP)** to standardize how agents interact with the outside world.

*   **Registry:** Tools are defined in `mcp.json`.
*   **Execution:** The `server/` acts as the MCP host.
*   **Workflow:**
    1.  Agent decides to use a tool (e.g., `fetch_stock_price`).
    2.  Agent emits a specific "Tool Call" token.
    3.  Orchestrator intercepts the token, calls the Python function in `server/`, and returns the result to the Agent's context window.

## 5. Technology Stack

*   **Language:** Python 3.10+ (Core), TypeScript (Frontend).
*   **Orchestration:** LangGraph (System 2), AsyncIO (System 1).
*   **Database:** Neo4j (Knowledge Graph), PostgreSQL (Transactional).
*   **LLM Interface:** LiteLLM (Abstraction layer for OpenAI, Anthropic, Local Models).
*   **Dependency Management:** `uv` (Rust-based pip alternative).

## 6. Security & Governance

*   **AOF (Agentic Oversight Framework):** A set of "Guardrails" that prevent agents from taking unauthorized actions (e.g., executing trades > $1M without human approval).
*   **Audit Trail:** Every thought and action is logged to `logfire` / Telemetry.
