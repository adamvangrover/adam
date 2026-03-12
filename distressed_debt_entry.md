
---

## `distressed_debt_agent`

*   **File:** `core/agents/specialized/distressed_debt_agent.py`
*   **Description:** A specialized underwriting agent designed for the Leveraged Finance and Distressed Debt market. It calculates deep credit risk, default probability, enterprise value in distressed scenarios, and restructuring strategies. Evaluates recovery rates across debt tranches and flags covenant tripwires.
*   **Configuration:** `config/agents.yaml`
    *   `model`: Typically defaults to `gemini-1.5-pro` or equivalent reasoning model.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.AgentBase`. Returns a strongly typed `DistressedDebtAnalysis` Pydantic model.
*   **Agent Forge and Lifecycle:** Created dynamically as part of the `credit_memo_orchestrator` flow or executed independently against a borrower profile.
*   **Model Context Protocol (MCP):** Requires `issuer_name`, `financials`, and `debt_structure` kwargs during `execute()`.
*   **Tools and Hooks:** None explicitly required, though relies on structured output generation.
*   **Compute and Resource Requirements:** Moderate (one complex LLM call per run). Includes deterministic fallback logic.
*   **Dependencies:** `pydantic`.
*   **Developer Notes:** Added as part of the System 2 credit enhancement protocol. Provides deterministic heuristic fallbacks if LLM fails, ensuring robust pipeline execution.
