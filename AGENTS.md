# Welcome to the ADAM Project!

This document provides guidance for AI agents working with the ADAM codebase.

## High-Level Goal

The primary goal of the ADAM project is to create a sophisticated, autonomous AI system that can perform complex financial analysis, generate insightful reports, and adapt to new information and user requirements.

## 🧭 Navigation & Context

*   **Core Engine Logic:** `core/AGENTS.md` - Strictly typed, System 2 reasoning.
*   **Web Dashboard:** `webapp/README_DEV.md` - Frontend development guidelines.
*   **Documentation:** `docs/` - User guides and architectural specs.

## Core Principles

When working on the ADAM project, please adhere to the following principles:

*   **Modularity:** Keep code modular and well-documented. Each component should have a clear purpose and interface.
*   **Extensibility:** Design components to be easily extended and adapted for new use cases.
*   **Robustness:** Implement comprehensive error handling and logging to ensure the system is resilient and debuggable.
*   **Efficiency:** Optimize code for performance, especially in data-intensive and computationally expensive tasks.
*   **Verify Your Work:** Always run tests or scripts to confirm your changes work as intended. **Never commit broken code.**

## Table of Contents

*   [High-Level Goal](#high-level-goal)
*   [Core Principles](#core-principles)
*   [System Architecture](#system-architecture)
*   [Directives for v26 Development](#directives-for-v26-development)
*   [Communication Protocols](#communication-protocols)
*   [Tool Usage (MCP)](#tool-usage-mcp)
*   [Testing Agents](#testing-agents)

## System Architecture

The ADAM system is built on a modular architecture that consists of several key components.

### Key Components

*   **Agents:** Autonomous agents that perform specific tasks, such as data retrieval, analysis, and reporting.
*   **Core:** The central infrastructure that supports the agents, including the main loop, data management, and communication.
*   **System 2 Engine:** Located in `core/engine/`, this graph-based system handles complex reasoning loops.

## Directives for v26 Development

### 1. The Prime Directive: Bifurcation
This repository implements two distinct strategies. **Do not mix them.**
- **Path A (`core/vertical_risk_agent`)**: Prioritize **Reliability**, **Auditability**, and **Business Logic**. Code should be defensive, heavily typed (Pydantic), and explained via logs. Performance is secondary to correctness.
- **Path B (`experimental/inference_lab`)**: Prioritize **Velocity**, **Throughput**, and **Math**. Code should be optimized (Triton/CUDA), minimal, and benchmarked. Business logic is irrelevant here.

### 2. Coding Standards

#### Path A (Product)
- **Imports**: Absolute imports preferred.
- **Error Handling**: Never crash. Use `try/except` blocks with logging.
- **State**: All agent state must be defined in `state.py` using `TypedDict` or `Pydantic`.
- **Tools**: All external interactions (API, DB) must go through the MCP Server pattern (`server/mcp_server.py`).

#### Path B (Research)
- **Imports**: Minimal overhead.
- **Memory**: Be conscious of VRAM. Use `inplace` operations where possible.
- **Comments**: Explain the *math* behind the optimization.

### 3. Workflow Protocol
1. **Plan**: Before writing code, inspect the `README.md` or `AGENTS.md` of the target directory.
2. **Visualize**: If modifying `langgraph` flows, generate the Mermaid diagram to verify logic.
3. **Verify**: Run `scripts/run_adam.py --mode test` (or relevant test script) before submitting.

### 4. Known Context
- The system uses a mock `langgraph` if the library is missing. Do not remove this fallback unless you are installing the real dependency.
- The `xbrl_handler.py` has a real XML parser but falls back to mock data if the file is missing. This is intentional for demo purposes.

## Communication Protocols

All agents must adhere to strict input/output contracts defined in `core/agents/templates/v26_template_agent.py`.

### Standard Input (`AgentInput`)
```python
class AgentInput(BaseModel):
    query: str = Field(..., description="The user's objective or question.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Shared graph state.")
```

### Standard Output (`AgentOutput`)
```python
class AgentOutput(BaseModel):
    answer: str = Field(..., description="The final synthesized answer.")
    sources: List[str] = Field(default_factory=list, description="List of citations.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Conviction score (0-1).")
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

**Rule:** Every agent response must include a `confidence` score. If confidence is < 0.85, the Consensus Engine may reject the output.

## Tool Usage (MCP)

Adam v26 uses the **Model Context Protocol (MCP)** to standardize how agents access external data and tools.

*   **Registry:** All available tools are defined in `mcp.json`.
*   **Server:** The tool execution logic resides in `server/server.py` and `server/mcp_server.py`.

### Adding a New Tool
1.  **Implement:** Add the Python function in `server/server.py` (or a dedicated module).
2.  **Register:** Update `mcp.json` with the tool name, description, and JSON schema.
3.  **Deploy:** Restart the MCP server to pick up changes.

### Calling Tools
Agents should not import tool functions directly. Instead, they should emit a "Tool Call" message which is intercepted by the `MetaOrchestrator` and routed to the MCP server.

## Testing Agents

Before committing a new agent, you must verify its behavior.

1.  **Unit Tests:** Create a test file in `tests/` (e.g., `tests/test_my_agent.py`).
2.  **Mocking:** Use `unittest.mock` to mock the MCP server responses. Do not make real API calls in unit tests.
3.  **Execution:** Run the test using pytest:
    ```bash
    PYTHONPATH=. pytest tests/test_my_agent.py
    ```
4.  **Integration:** Use the CLI for a live test:
    ```bash
    python scripts/run_adam.py --agent "MyNewAgent" --query "Test query"
    ```

Thank you for your contributions to the ADAM project!
