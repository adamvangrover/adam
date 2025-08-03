# Sub-Agents Development Guide

This document provides guidelines and best practices for developing and maintaining Sub-Agents within the CreditSentry ecosystem.

## Role and Philosophy

Sub-Agents are the "worker bees" of the system. They are the foundational layer responsible for interacting directly with data sources and tools. Their primary purpose is to perform specific, narrow, and well-defined tasks related to data acquisition and processing.

**Core Principles:**

*   **Do One Thing Well:** Each Sub-Agent should have a single, clear responsibility (e.g., fetch data from one specific API, parse one type of document). Avoid creating monolithic Sub-Agents that handle multiple, unrelated tasks.
*   **Produce Structured, Verifiable Data:** The output of a Sub-Agent must always be in a structured format (e.g., JSON) and must adhere to the system-wide metadata schema. This includes providing a `source_agent`, `confidence_score`, and other critical metadata.
*   **Be Tool-Users, Not Thinkers:** Sub-Agents should focus on executing tasks using their tools (e.g., OCR engines, API clients). They should not perform complex analysis, synthesis, or decision-making. That is the role of Meta-Agents.
*   **Fail Gracefully:** Sub-Agents must be robust and handle errors gracefully. If a data source is unavailable or a tool fails, the agent should log the error clearly and return a structured error message, not crash.

## Adding a New Sub-Agent

To add a new Sub-Agent, follow these steps:

1.  **Create the Agent File:** Create a new Python file in this directory (e.g., `my_new_sub_agent.py`).
2.  **Define the Class:** Create a new class that inherits from `core.agents.agent_base.AgentBase`.
    ```python
    from core.agents.agent_base import AgentBase

    class MyNewSubAgent(AgentBase):
        # ...
    ```
3.  **Implement the `__init__` method:** Initialize the agent, including any tools or clients it will need.
4.  **Implement the `execute` method:** This is the main entry point for the agent. It should contain the core logic for the agent's task. The method signature should be clearly defined to accept the necessary inputs.
5.  **Register the Agent:** Add the new agent to the `config/agents.yaml` file under the `credit_sentry_agents.sub_agents` section. Provide a clear `persona`, `description`, and `expertise`.
6.  **Write Unit Tests:** Create a corresponding test file in the `tests/` directory to ensure your agent functions correctly and handles edge cases.

## Best Practices

*   **Configuration:** All external endpoints, API keys, and other configuration should be passed in via the `config` dictionary in the constructor, not hard-coded in the agent.
*   **Logging:** Use the standard Python `logging` module to log important events, warnings, and errors. This is crucial for debugging and monitoring.
*   **Code Comments:** Document your code clearly, especially the `execute` method's inputs, outputs, and core logic.
*   **Metadata Compliance:** Ensure the final output of your agent strictly follows the standard data object schema defined in the Orchestrator's meta-prompt. This is essential for the integrity of the entire system.
