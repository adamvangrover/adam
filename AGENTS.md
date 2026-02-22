# The Adam Agent Developer's Bible

> **"Code defines the body; Prompts define the mind."**

This document is the absolute source of truth for creating, modifying, and debugging agents within the Adam v26.0 ecosystem. Deviating from these standards will result in PR rejection.

---

## 🧭 Navigation

*   [The Prime Directive: Bifurcation](#the-prime-directive-bifurcation)
*   [Pheromones & Memory Hooks](#pheromones--memory-hooks-mandatory)
*   [Architecture: System 1 vs. System 2](#architecture-system-1-vs-system-2)
*   [Communication Protocols](#communication-protocols)
*   [Debugging & Tracing](#debugging--tracing)
*   [Best Practices](#best-practices)
*   [Swarm Protocols](#swarm-protocols--inter-agent-communication)

---

## The Prime Directive: Bifurcation

This repository supports two conflicting goals: **Reliability** and **Velocity**. To manage this, we strictly bifurcate the codebase.

### Path A: The Product (Reliability)
*   **Locations:** `core/agents/`, `core/credit_sentinel/`
*   **Philosophy:** "Slow is Smooth, Smooth is Fast."
*   **Specialized Mandates:**
    *   **Fundamental Agent:** Forward-looking cash flow modeling, Enterprise Value (EV), and Base/Bull/Bear scenarios.
    *   **Risk Agent:** Probability of Default, Liquidity Runway (Months), and Recovery Rate (% of Assets).
    *   **Credit Sentinel:** Continuous monitoring via System 1 Swarm for covenant breaches and distress signals.
*   **Requirements:**
    *   **Strict Typing:** Pydantic models for ALL inputs/outputs.
    *   **Defensive Coding:** `try/except` blocks around every external call.
    *   **Auditability:** Every decision must be logged with a reasoning trace.
    *   **No Magic Numbers:** All constants must be in a config file.

### Path B: The Lab (Velocity)
*   **Locations:** `experimental/`, `research/`, `tinker_lab/`
*   **Philosophy:** "Move Fast and Break Things."
*   **Requirements:**
    *   **Minimal Overhead:** Raw dictionaries are fine.
    *   **Optimization:** Focus on VRAM usage and token throughput.
    *   **Experimentation:** Feel free to monkey-patch or use bleeding-edge libraries.

**CRITICAL RULE:** Do not import "Lab" code into "Product" modules. The Product must be stable.

---

## Pheromones & Memory Hooks (MANDATORY)

**"Those who cannot remember the past are condemned to repeat it."**

Before writing a single line of code, you **MUST** consult the **[Agent Knowledge Base](docs/AGENTS_KNOWLEDGE_BASE.md)**.

This document aggregates critical lessons from Sentinel (Security), Bolt (Architecture), Palette (UX), and the Swarm (Async).

### The Pre-Flight Checklist
1.  **Security Check:** Are you handling user input? (Check `docs/AGENTS_KNOWLEDGE_BASE.md#security`)
2.  **Graph Check:** Are you modifying the Knowledge Graph? (Check `docs/AGENTS_KNOWLEDGE_BASE.md#architecture`)
3.  **Frontend Check:** Are you touching the UI? (Check `docs/AGENTS_KNOWLEDGE_BASE.md#ux`)
4.  **Logging:** If you discover a new "trap", log it in `docs/AGENTS_KNOWLEDGE_BASE.md` immediately.

---

## Architecture: System 1 vs. System 2

Understanding where your agent fits is crucial.

### System 1: The Swarm (Async)
*   **Pattern:** Fire-and-Forget.
*   **Base Class:** `AsyncAgentBase`
*   **Use Case:** Fetching data, monitoring news feeds, simple classification.
*   **Example:** `NewsWatcherAgent` sees a headline and pushes it to the queue.

### System 2: The Graph (Sync)
*   **Pattern:** State Machine (LangGraph).
*   **Base Class:** `TemplateAgentV26`
*   **Use Case:** Complex reasoning, multi-step planning, report generation.
*   **Example:** `FundamentalAnalyst` receives a ticker, plans a research path, fetches 10-Ks, analyzes them, and writes a memo.

---

## Communication Protocols

All "System 2" agents must adhere to the **Standard Interface**.

### 1. Input Schema (`AgentInput`)

```python
from pydantic import BaseModel, Field
from typing import Dict, Any

class AgentInput(BaseModel):
    query: str = Field(..., description="The specific question or objective.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Shared graph state (RAG data, previous results).")
    tools: List[str] = Field(default_factory=list, description="List of allowed tool names.")
```

### 2. Output Schema (`AgentOutput`)

```python
class AgentOutput(BaseModel):
    answer: str = Field(..., description="The final synthesized answer.")
    sources: List[str] = Field(default_factory=list, description="List of citations (filenames, URLs).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Conviction score (0.0 to 1.0).")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Debug info, token usage, etc.")
```

**Constraint:** If `confidence` < 0.85, the Consensus Engine will flag the result for human review.

---

## Debugging & Tracing

Agents are complex. Use these tools to diagnose issues.

### 1. The `--debug` Flag
Running the main script with `--debug` enables verbose logging for the Planner and Agents.

```bash
python scripts/run_adam.py --mode deep_dive --ticker AAPL --debug
```

### 2. Trace Analysis
Look for the standard log prefixes:
*   `[Planner]`: High-level goal decomposition.
*   `[Orchestrator]`: Routing decisions.
*   `[Agent:Risk]`: Specific agent logs.

### 3. Common Errors
*   **"Hallucinated Tool":** The agent tried to call a tool that isn't in `mcp.json`.
    *   *Fix:* Check the system prompt tool definitions.
*   **"Low Conviction":** The agent returned a generic answer because it couldn't find data.
    *   *Fix:* Check the `context` passed to the agent. Was the RAG retrieval successful?

---

## Best Practices

1.  **Grounding:** Every claim needs a source. If you calculate a ratio, cite the line item in the 10-K.
2.  **Tool Use:** Agents should prefer using Tools (Python functions) over doing math in their head.
    *   *Bad:* LLM tries to calculate `EBITDA / Interest`.
    *   *Good:* LLM calls `calculate_ratio(ebitda, interest)`.
3.  **Prompt Versioning:** Do not hardcode prompts. Load them from `prompt_library/AOPL-v2.0/`.
4.  **Testing:** Write a unit test for every new agent using `pytest`. Mock the LLM response to test logic flow.

---

## Tool Usage (MCP)

We use the **Model Context Protocol (MCP)**.
To add a tool:
1.  Define the function in `server/server.py`.
2.  Register it in `mcp.json`.
3.  Restart the server.

Refer to `docs/architecture.md` for more details.

---

## Swarm Protocols & Inter-Agent Communication

To prevent "Graph Spaghetti" and circular dependencies, follow these rules:

1.  **No Direct Calls**:
    *   Do NOT instantiate `AgentB` inside `AgentA`.
    *   *Bad:* `result = RiskAgent().execute(...)` inside `LegalAgent`.
    *   *Good:* Return a request to the `MetaOrchestrator` to run `RiskAgent` next.

2.  **Use the Orchestrator**:
    *   If an agent needs info from another domain, it should signal this in its output `metadata`.
    *   Example: `metadata={"next_step": "consult_legal", "query": "Check covenants"}`.

3.  **Conflict Resolution**:
    *   If two agents (e.g., Risk and Growth) provide conflicting advice, the `ConsensusEngine` will arbitrate based on their `confidence` scores and the user's risk profile.
