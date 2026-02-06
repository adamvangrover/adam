# Adam Agents Registry

This directory contains the specialized intelligence units (Agents) that power the Adam system.

## 🧠 Agent Taxonomy

Adam v26.0 distinguishes between three types of agents:

### 1. Specialized Agents (Workers)
Located in `core/agents/specialized/`. These agents possess deep domain expertise but narrow scope.
*   **Examples:** `FundamentalAnalyst`, `RiskAnalyst`, `LegalSentinel`.
*   **Architecture:** Typically implemented as a subgraph in LangGraph or a specialized tool user.

### 2. Meta Agents (Managers)
Located in `core/agents/meta_agents/`. These agents coordinate other agents or perform higher-order reasoning.
*   **Examples:** `MetaCognitiveAgent`, `DiscussionChairAgent`.
*   **Architecture:** Orchestrators that route tasks and evaluate outputs.

### 3. Swarm Agents (Async)
Located in `core/system/v22_async/` (and referenced here). These are high-throughput, stateless workers for data fetching and monitoring.

## 📂 Directory Structure

| Directory | Description |
| :--- | :--- |
| `specialized/` | Domain experts (Finance, Law, Tech). |
| `meta_agents/` | Managers and coordinators. |
| `templates/` | Reference implementations for new agents. |
| `sub_agents/` | Smaller, utility-focused units. |

## 🛠️ Developing New Agents

To create a new agent:
1.  Copy `core/agents/templates/v26_template_agent.py`.
2.  Define your Input/Output state using Pydantic.
3.  Implement the `execute` logic.
4.  Register the agent in `core/engine/neuro_symbolic_planner.py`.

See `docs/agent_development.md` for a full walkthrough.

## 🚦 Quick Reference: Dos and Don'ts

| Category | Do ✅ | Don't ❌ |
| :--- | :--- | :--- |
| **Communication** | Route requests via `MetaOrchestrator`. | Instantiate other agents directly (`RiskAgent().execute()`). |
| **I/O** | Use strict Pydantic models. | Pass raw Dictionaries or JSON strings. |
| **Tools** | Use `mcp.tool()` definitions. | Hardcode API calls inside the agent logic. |
| **Grounding** | Cite sources in `AgentOutput.sources`. | Hallucinate facts or omit citations. |
| **Config** | Load constants from `config/`. | Magic numbers in code (e.g., `threshold = 0.8`). |

*Refer to the [Agent Developer Bible](../../AGENTS.md) for detailed protocols.*
