# Meta-Agents Development Guide

This document provides guidelines and best practices for developing and maintaining Meta-Agents within the CreditSentry ecosystem.

## Role and Philosophy

Meta-Agents are the "analysts" and "strategists" of the system. They represent the cognitive core, responsible for performing higher-order tasks that require analysis, synthesis, and interpretation. They do not interact directly with external data sources; instead, they operate on the structured, verified data provided by Sub-Agents.

**Core Principles:**

*   **Synthesize, Don't Gather:** The primary role of a Meta-Agent is to take structured data from one or more Sub-Agents and transform it into a more abstract or analytical form (e.g., a risk rating, a summary, a forecast).
*   **Trust but Verify Metadata:** Meta-Agents should trust the data provided by Sub-Agents but must be programmed to inspect and act upon the metadata. For example, a Meta-Agent should handle data with a low `confidence_score` differently, perhaps by flagging its own output as being based on uncertain data.
*   **Encapsulate Complex Logic:** Complex business logic, analytical models, and qualitative frameworks (like the "5 Cs of Credit") should be encapsulated within Meta-Agents.
*   **Maintain State (If Necessary):** Some Meta-Agents, like the `PortfolioMonitoringEWSAgent`, may need to maintain state over time (e.g., tracking trends). This should be done carefully and with clear documentation.

## Adding a New Meta-Agent

To add a new Meta-Agent, follow these steps:

1.  **Create the Agent File:** Create a new Python file in this directory (e.g., `my_new_meta_agent.py`).
2.  **Define the Class:** Create a new class that inherits from `core.agents.agent_base.AgentBase`.
3.  **Implement the `__init__` method:** Initialize the agent.
4.  **Implement the `execute` method:** This is the main entry point. Its inputs will typically be the structured outputs from one or more Sub-Agents. The method should contain the core analytical or synthesis logic.
5.  **Register the Agent:** Add the new agent to the `config/agents.yaml` file under the `credit_sentry_agents.meta_agents` section. Provide a clear `persona`, `description`, and `expertise`.
6.  **Write Unit Tests:** Create corresponding tests to validate the agent's analytical logic and its handling of various input data scenarios.

## Best Practices

*   **Input Validation:** The `execute` method should validate its input to ensure it has received the necessary data structures from the Sub-Agents before proceeding with its analysis.
*   **Clarity of Logic:** The analytical logic within a Meta-Agent should be well-documented and, where possible, broken down into smaller, private helper methods to improve readability and maintainability.
*   **Model Integration:** If a Meta-Agent uses a machine learning model, the model should be loaded and managed according to the project's Model Risk Management (MRM) guidelines.
*   **Output Schema:** The output of a Meta-Agent should also be a structured object, though its schema will be more analytical than the raw data from Sub-Agents. It must still contain appropriate metadata.
