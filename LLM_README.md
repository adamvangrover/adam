# Adam v26.0: The Neuro-Symbolic Financial Sovereign (LLM Context Optimizer)

> **FOR LLMS:** This file is your "System Prompt Extension". It aggregates the most critical context from `README.md`, `AGENTS.md`, and `docs/` to help you understand the codebase instantly.

---

## 1. Project Identity & Philosophy

**Adam v26.0** is an autonomous financial analysis system designed to solve the "Epistemological Crisis" of LLM hallucinations in finance. It fuses two cognitive architectures:

1.  **System 1 (The Swarm):**
    *   **Role:** Perception & Reflexes. Fast, asynchronous, non-blocking.
    *   **Implementation:** `core/system/v22_async/` (Pub/Sub, Event Loop).
    *   **Use Case:** News ingestion, sentiment scoring, data fetching.
2.  **System 2 (The Graph):**
    *   **Role:** Reasoning & Planning. Slow, synchronous, stateful.
    *   **Implementation:** `core/engine/` (Directed Acyclic Graph / DAG).
    *   **Use Case:** Deep dive analysis, risk modeling, credit memo writing.

**Core Directive:** "Slow is Smooth, Smooth is Fast." Reliability > Velocity in production paths.

---

## 2. High-Level Architecture

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

---

## 3. Directory Map (The "Brain")

*   **`core/`**: The cognitive center.
    *   `agents/`: The workforce (Specialized & Meta agents).
    *   `engine/`: The control center (Planner, Orchestrator, Consensus).
    *   `credit_sentinel/`: Domain expert for Distressed Debt (ICAT, Ratio Calculator).
    *   `system/`: Infrastructure (Swarm, Memory, Context).
    *   `data_processing/`: Universal Ingestor (ETL pipeline).
*   **`services/webapp/`**: The "Neural Dashboard" (React/Flask).
    *   Visualizes the "Thought Process" via `AgentIntercom` (SSE).
*   **`server/`**: The MCP (Model Context Protocol) Server.
    *   Exposes Python tools to the agents.
*   **`prompt_library/`**: The "Mind" (AOPL v26.0 Prompts).
    *   Prompts are treated as code, versioned and structured.

---

## 4. Critical Directives (The "Rules of Engagement")

### A. The Bifurcation Protocol
*   **Path A: The Product (`core/agents`, `core/credit_sentinel`)**
    *   **MUST:** Be strictly typed (Pydantic), defensive (`try/except`), and auditable.
    *   **MUST:** Use `core.security.safe_unpickler` instead of `pickle`.
*   **Path B: The Lab (`experimental/`, `research/`)**
    *   **ALLOWED:** Rapid prototyping, loose schemas.
    *   **FORBIDDEN:** Importing Lab code into Product modules.

### B. Security Mandates (P0 Risks)
1.  **NO `pickle.load()`**: Use `core.security.safe_unpickler.safe_load()`.
2.  **NO Dynamic Imports**: Do not use `importlib` with user input.
3.  **NO `eval()` / `exec()`**: Strict prohibition on executing arbitrary code.
4.  **Sanitize Inputs**: Validate all external data (URLs, Files) before processing.

### C. Orchestration Rules
1.  **NO Direct Calls**: `AgentA` must **never** instantiate `AgentB` directly.
2.  **Use the Orchestrator**: Return a request to the `MetaOrchestrator` to schedule `AgentB`.
3.  **State Isolation**: Agents should not mutate shared global state directly; use the `ContextManager`.

---

## 5. Agent Roster (Key Personnel)

| Agent Name | Role | Location |
| :--- | :--- | :--- |
| **Meta Orchestrator** | Routes tasks, manages Swarm/Graph handoffs. | `core/engine/orchestrator.py` |
| **Neuro-Symbolic Planner** | Decomposes goals into execution graphs. | `core/engine/planner.py` |
| **Credit Sentinel** | Distressed debt analysis (ICAT, Covenants). | `core/credit_sentinel/` |
| **Risk Analyst** | Quantitative & Qualitative risk assessment. | `core/agents/risk_assessment_agent.py` |
| **Fundamental Analyst** | Deep dive into 10-K/10-Q filings. | `core/agents/fundamental_analyst_agent.py` |
| **Legal Sentinel** | Reviews contracts & regulatory compliance. | `core/agents/legal_agent.py` |
| **Repo Guardian** | CI/CD gatekeeper, reviews PRs. | `core/agents/governance/repo_guardian/` |
| **Chronos Agent** | Manages temporal context & memory. | `core/agents/meta_agents/chronos_agent.py` |

---

## 6. Data Flow & Lifecycle

1.  **Ingestion (System 1):**
    *   `UniversalIngestor` reads raw files/URLs.
    *   Data is cleaned, chunked, and vectorized (`core/data_processing`).
    *   "Semantic Conviction" scores reliability.
2.  **Planning (System 2):**
    *   `MetaOrchestrator` receives query.
    *   `Planner` builds a DAG of tasks.
3.  **Execution (Agents):**
    *   Agents execute tasks using Tools (via MCP).
    *   Results are stored in `GlobalContext`.
4.  **Synthesis (Consensus):**
    *   `ConsensusEngine` aggregates outputs.
    *   If confidence < 85%, loop back to Planner (Reflexion).
5.  **Output:**
    *   Final report generated (Markdown/PDF).
    *   "Thought Trace" streamed to WebApp.

---

## 7. Setup & Tooling

*   **Dependency Management:** `uv` (Rust-based).
    *   `uv sync`: Install dependencies.
*   **Tool Protocol:** MCP (Model Context Protocol).
    *   Tools defined in `mcp.json`.
    *   Server runs in `server/`.

---

## 8. How to Navigate This Repo (For LLMs)

*   **To Understand Logic:** Read `core/agents/` and `core/engine/`.
*   **To Understand Rules:** Read `AGENTS.md` and `docs/AGENTS_KNOWLEDGE_BASE.md` (MANDATORY).
*   **To Understand Tools:** Read `server/server.py` and `mcp.json`.
*   **To Understand Data:** Read `core/data_processing/`.
*   **To Understand UI:** Read `services/webapp/`.

**Remember:** You are an autonomous engineer working on a high-stakes financial system. Precision, security, and reliability are your watchwords.
