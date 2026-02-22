# Adam v26.0 Architecture: The Neuro-Symbolic Sovereign

## 1. Executive Overview

Adam v26.0 is not a chatbot. It is an **Institutional-Grade Cognitive Engine** designed for high-stakes financial analysis, risk modeling, and strategic decision-making.

The architecture is built on the **System 1 / System 2** cognitive theory:
*   **System 1 (The Swarm):** Fast, intuitive, parallel, and asynchronous. Handles perception (news ingestion), reflexive actions (alerts), and massive data processing.
*   **System 2 (The Graph):** Slow, deliberate, logical, and sequential. Handles deep reasoning, complex planning, valuation modeling, and final adjudication.

## 2. Core Architectural Pillars

### 2.1 The Neuro-Symbolic Hybrid
Adam fuses two distinct AI paradigms:
*   **Neural (LLMs):** Used for semantic understanding, creativity, and qualitative analysis (e.g., "Read this MD&A and gauge management sentiment").
*   **Symbolic (Graphs/Code):** Used for deterministic logic, strict math, and structural constraints (e.g., "Calculate DCF where WACC = 8.5%").

**Implementation:**
*   **LangGraph:** Manages the state machine of the "System 2" reasoning loops.
*   **Pydantic:** Enforces strict data schemas (the "Symbolic" guardrails) at every step.

### 2.2 The Bifurcation Protocol
To balance stability with innovation, the codebase is strictly split:
*   **Product Path (`core/agents`, `core/engine`):** Production-grade, strictly typed, rigorously tested.
*   **Lab Path (`experimental/`, `tinker_lab`):** Rapid prototyping, loose schemas, bleeding-edge features.

**Rule:** Lab code *never* imports into Product code without a formal graduation process.

### 2.3 Logic as Data
Business logic is decoupled from application code.
*   **Rules Engine:** Risk thresholds and trading triggers are stored in JSON/YAML configuration files.
*   **Benefit:** Adjusting a risk limit from 15% to 12% does not require a code deploy, only a config update.

## 3. System Components

### 3.1 The Meta-Orchestrator (`core/engine/meta_orchestrator.py`)
The "Brain" of the system.
1.  **Receives Query:** "Analyze AAPL debt."
2.  **Assesses Complexity:**
    *   *Low:* Route to simple tool (e.g., `fetch_price`).
    *   *High:* Route to Neuro-Symbolic Planner.
3.  **Dispatches:** Activates the appropriate Graph or Swarm.

### 3.2 The Swarm (System 1)
*   **HiveMind:** A manager for asynchronous worker agents.
*   **Workers:** Specialized, lightweight tasks (e.g., `NewsScanner`, `SentimentScorer`).
*   **Communication:** Pub/Sub event bus (internal or Redis-backed).

### 3.3 The Graph (System 2)
*   **Deep Dive Graph:** A 5-phase sequential DAG for comprehensive analysis.
    1.  **Entity Resolution:** Who is this?
    2.  **Fundamentals:** What are the numbers?
    3.  **Credit:** Will they go bankrupt?
    4.  **Risk:** What if the world breaks?
    5.  **Synthesis:** Buy or Sell?
*   **State Management:** Each graph run maintains a strictly typed `State` object (e.g., `OmniscientState`), ensuring data continuity.

## 4. Memory Systems

*   **Short-Term (Context):** In-memory context window of the LLM.
*   **Medium-Term (Graph State):** The `State` object passed between nodes in a LangGraph workflow.
*   **Long-Term (Vector Store):** Embeddings of past reports, news, and 10-Ks, retrievable via RAG (Retrieval-Augmented Generation).
*   **Episodic (Logs):** `SystemLogger` and `ProofOfThoughtLogger` record every decision trace for auditability.

## 5. Security & Governance

*   **AOF (Agentic Oversight Framework):** "Guardrails" that prevent unauthorized actions.
*   **Sandboxing:** Code execution (e.g., `CodeAlchemist`) runs in isolated environments (Docker/GVisor recommended for production).
*   **Audit Trail:** Every "Tool Call" and "Strategic Verdict" is hashed and logged.
