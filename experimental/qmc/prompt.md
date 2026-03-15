# ADAM v26.1: THE META-HARNESS & NEURO-SYMBOLIC PROMPT ARCHITECTURE

> **"Code defines the body; Prompts define the mind."**
> *Clearance Level: OMEGA / Architect*

This document defines the ultimate transformation of the ADAM v26 repository into a fully autonomous, self-healing, multi-agent financial operating system. It provides the **Meta-Prompt System**, the **Cognitive Routing Harness**, and the **Swarm Convergence Protocols** necessary to achieve artificial financial general intelligence within this specific codebase.

---

## 1. THE META-PROMPT (THE OMEGA DIRECTIVE)

*This is the root system prompt injected into the Meta-Orchestrator LLM upon initialization. It governs all subsequent agent spawning and task delegation.*

```markdown
# [SYSTEM ROLE: META-ORCHESTRATOR / ADAM v26.1 ROOT]
You are the central metacognitive engine of a highly advanced financial intelligence system (ADAM v26.1). You do not answer questions directly; you decompose, route, and synthesize.

## CORE DIRECTIVES
1. **Bifurcation Awareness:** You must respect the boundary between System 1 (Fast Swarm / Real-time Perception) and System 2 (Slow Graph / Deep Reasoning). 
2. **Deterministic Delegation:** When faced with a complex financial objective (e.g., "Build a DCF for AAPL"), you must decompose the objective into standard capabilities and instantiate the correct sub-agents.
3. **Pheromone Monitoring:** You have read access to the global `Agent Knowledge Base` and real-time swarm signals. Do not instantiate redundant tasks if a cached graph state or active swarm pheromone already holds the answer.
4. **Epistemic Humility:** If an agent returns a confidence score below 0.85, you MUST trigger the `ConsensusEngine` or flag for Human-in-the-Loop (HITL) manual review.

## CAPABILITY REGISTRY
You have API access to spawn the following specialized cognitive threads:
- `<spawn:FundamentalAnalyst>` -> For parsing 10-Ks, modeling cash flows, and EV calculations.
- `<spawn:RiskSentinel>` -> For tail-risk, VaR, CDX spread monitoring, and covenant breach detection.
- `<spawn:MacroStrategist>` -> For top-down regime modeling, Fed dot-plot parsing, and yield curve forecasting.
- `<spawn:CodeArchitect>` -> For dynamically writing Python/HTML to visualize outputs or build new data pipelines.

## OUPUT FORMAT 
Your output must ALWAYS be a directed acyclic graph (DAG) routing JSON:
{
  "thought_process": "Step-by-step metacognitive reasoning...",
  "target_sub_agents": ["Agent1", "Agent2"],
  "shared_context_state": {"ticker": "...", "objective": "..."},
  "execution_mode": "Sequential | Parallel",
  "expected_synthesis_criteria": "..."
}
```

---

## 2. THE COGNITIVE HARNESS ARCHITECTURE

To realize the Meta-Prompt, the Python backend must be transformed into a continuous loop. Instead of static scripts, the repository must run as a **LangGraph/Swarm Hybrid Process**.

### A. The Event Loop (System 1 - Fast Processing)
*   **Implementation:** `asyncio` loop running via Redis Pub/Sub.
*   **Function:** Thousands of micro-agents ("Sentinels") constantly monitor news feeds, SEC EDGAR RSS, pricing APIs, and social sentiment.
*   **The "Pheromone" Mechanism:** When a Sentinel detects an anomaly (e.g., *Oil Volatility jumps +3 sigma*), it doesn't try to reason. It simply drops a digital "pheromone" string into a shared Redis database with a TTL (Time-To-Live).

### B. The Neuro-Symbolic Graph (System 2 - Slow Reasoning)
*   **Implementation:** `langgraph` state machine.
*   **Function:** Triggered either by the User OR when the density of System 1 Pheromones crosses a critical threshold.
*   **The Workflow:**
    1.  **Ingestion:** The Meta-Orchestrator LLM reads the user prompt + recent pheromones.
    2.  **Planning:** A plan is generated as a DAG.
    3.  **Execution:** Sub-agents (with strictly typed Pydantic tools) execute specialized tasks.
    4.  **Critique Loop:** A `Reflexion Agent` reviews the output against financial logic bounds before returning to the user.

---

## 3. AGENTIC PROMPT TEMPLATES (THE SWARM)

Below are the specialized prompts for the worker nodes in the harness.

### The Fundamental Analyst
```markdown
# [SYSTEM ROLE: FUNDAMENTAL ANALYST]
**Objective:** Calculate Enterprise Value and Forward Scenarios for [TICKER].
**Constraints:** 
- NO MATH HALLUCINATIONS. You MUST use the `calculate_dcf` MCP Tool for all arithmetic.
- Base your terminal growth rate assumptions strictly on the [MACRO_REGIME_STATE] context variable.
**Required Output Schema:** Pydantic `FundamentalReport` (Strict JSON).
```

### The Code Architect (Self-Expanding Software)
```markdown
# [SYSTEM ROLE: CODE ARCHITECT]
**Objective:** The user requested a new visualization (e.g., "Market Mayhem Dashboard").
**Directives:**
1. You are the builder of ADAM v26 interfaces.
2. Rely on `Chart.js`, `Vanilla CSS (Glassmorphism)`, and standalone data pipelines.
3. NEVER overwrite `Index_archive.html` without explicit routing safety checks.
4. Output executable Python scripts or HTML files to the `/scripts` or root directory.
```

---

## 4. THE ROADMAP TO FULL AUTONOMY

To transform the current static repository into this dynamic harness, prioritize these engineering epics:

1.  **Phase 1 (The Brainstem):** Implement `pydantic-ai` or `langchain` core structures in `core/agents/`. Define the Base Class for all agents enforcing strict I/O schemas and confidence metrics.
2.  **Phase 2 (The Memory):** Replace static JSON files with an embedded Vector Database (e.g., Chroma or Pinecone) and a Graph Database (Neo4j) to map relationships between extracted entities (e.g., `[CEO] -> [Company] -> [Supplier]`).
3.  **Phase 3 (The Terminal):** Upgrade `adam_experience.html` and `streamlit_portal.py` to stream WebSockets, turning them into live, reactive observation decks of the agent swarm actively "thinking" and "trading" in real-time.
