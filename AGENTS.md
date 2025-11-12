# Welcome to the ADAM Project!

This document provides guidance for AI agents working with the ADAM codebase.

## High-Level Goal

The primary goal of the ADAM project is to create a sophisticated, autonomous AI system that can perform complex financial analysis, generate insightful reports, and adapt to new information and user requirements.

## Core Principles

When working on the ADAM project, please adhere to the following principles:

*   **Modularity:** Keep code modular and well-documented. Each component should have a clear purpose and interface.
*   **Extensibility:** Design components to be easily extended and adapted for new use cases.
*   **Robustness:** Implement comprehensive error handling and logging to ensure the system is resilient and debuggable.
*   **Efficiency:** Optimize code for performance, especially in data-intensive and computationally expensive tasks.

## Table of Contents

*   [High-Level Goal](#high-level-goal)
*   [Core Principles](#core-principles)
*   [System Architecture](#system-architecture)
    *   [Key Components](#key-components)
    *   [Component Interaction](#component-interaction)
*   [Agent Architecture](#agent-architecture)
    *   [Sub-Agents](#sub-agents)
    *   [Meta-Agents](#meta-agents)
    *   [Orchestrator Agents](#orchestrator-agents)
*   [Getting Started](#getting-started)
*   [Contribution Guidelines](#contribution-guidelines)

## System Architecture

The ADAM system is built on a modular architecture that consists of several key components. These components work together to provide a flexible and extensible platform for building and deploying autonomous agents.

### Key Components

*   **Agents:** Autonomous agents that perform specific tasks, such as data retrieval, analysis, and reporting. Agents are the core of the ADAM system and are responsible for carrying out the main logic of the application.
*   **Core:** The central infrastructure that supports the agents, including the main loop, data management, and communication. The core provides the essential services that agents need to operate, such as a message bus for inter-agent communication and a data store for persisting information.
*   **Data Sources:** Modules for accessing various data sources, such as APIs and databases. Data sources provide a standardized interface for retrieving data, regardless of the underlying source.
*   **LLM Engine:** The language model engine that provides natural language processing capabilities. The LLM engine is used for tasks such as text generation, summarization, and question answering.
*   **Simulations:** Environments for testing and evaluating the agents' performance. Simulations provide a controlled environment for running experiments and measuring key performance indicators.

### Component Interaction

The components of the ADAM system interact with each other in the following way:

1.  The **core** initializes the system and starts the main loop.
2.  The **core** loads the **agents** and other components based on the configuration files.
3.  **Agents** use the **data sources** to retrieve data from various sources.
4.  **Agents** use the **LLM engine** to perform natural language processing tasks.
5.  **Agents** communicate with each other through the **core's** message bus.
6.  **Simulations** use the **core** to run experiments and evaluate the performance of the **agents**.

## Agent Architecture

The ADAM system employs a hierarchical agent architecture that consists of three types of agents: Sub-Agents, Meta-Agents, and Orchestrator Agents. This architecture is designed to separate concerns, improve modularity, and enable the development of sophisticated AI applications.

### Sub-Agents

*   **Role:** Sub-Agents are the "worker bees" of the system. They are responsible for performing specific, narrow, and well-defined tasks related to data acquisition and processing.
*   **Responsibilities:**
    *   Interacting directly with data sources and tools (e.g., APIs, databases, OCR engines).
    *   Producing structured, verifiable data with metadata (e.g., source, confidence score).
    *   Handling errors gracefully and providing structured error messages.
*   **Example:** A Sub-Agent might be responsible for fetching financial data from a specific API or parsing a specific type of document.

### Meta-Agents

*   **Role:** Meta-Agents are the "analysts" and "strategists" of the system. They are responsible for performing higher-order tasks that require analysis, synthesis, and interpretation.
*   **Responsibilities:**
    *   Operating on the structured, verified data provided by Sub-Agents.
    *   Encapsulating complex business logic, analytical models, and qualitative frameworks.
    *   Transforming data into a more abstract or analytical form (e.g., a risk rating, a summary, a forecast).
*   **Example:** A Meta-Agent might take financial data from a Sub-Agent and use it to generate a credit risk assessment.

#### New Meta-Agents in v21.0

*   **Behavioral Economics Agent:** Analyzes market data and user interactions for signs of cognitive biases and irrational behavior.
*   **Meta-Cognitive Agent:** Monitors the reasoning and outputs of other agents to ensure logical consistency, coherence, and alignment with core principles.

### Orchestrator Agents

*   **Role:** The Orchestrator Agent is the "brain" and "central nervous system" of the entire agentic system. It is the highest level of control, responsible for interpreting user intent, formulating plans, delegating tasks, and synthesizing a final, coherent response.
*   **Responsibilities:**
    *   Understanding user goals and decomposing them into a sequence of tasks.
    *   Delegating tasks to the appropriate Sub-Agents and Meta-Agents.
    *   Managing the workflow, including handling dependencies and errors.
    *   Synthesizing the results from multiple agents into a final response.
*   **Example:** An Orchestrator Agent might take a user query like "What is the credit risk of Apple Inc.?", delegate the task of gathering financial data to a Sub-Agent, delegate the task of assessing credit risk to a Meta-Agent, and then synthesize the results into a comprehensive report.

### Hybrid Architecture (v22)

The ADAM system has been updated to a hybrid architecture that combines the synchronous, centrally-orchestrated model of v21 with the new asynchronous, message-driven model of v22. This dual-architecture design allows the system to leverage the strengths of both approaches, providing flexibility and scalability while maintaining the robustness of the original system.

The `HybridOrchestrator` is the central component of the new architecture. It acts as a bridge between the synchronous and asynchronous subsystems, providing a single entry point for all workflow execution. The `HybridOrchestrator` inspects each workflow to determine whether it is synchronous or asynchronous and then delegates it to the appropriate manager. For more details, see `docs/v22_architecture_integration.md`.

### Adaptive Architecture (v23)

The system is currently evolving towards the v23 "Adaptive System" architecture. This next-generation model introduces a stateful, cyclical graph-based execution engine (leveraging LangGraph) to enable true adaptive intelligence, including iterative self-correction, neuro-symbolic planning, and multimodal perception.

The `MetaOrchestrator` is the unified entry point for all execution models, including v21 (synchronous), v22 (asynchronous), and v23 (graph-based). For the complete vision and technical specifications, please refer to the official `docs/v23_architecture_vision.md` mandate.

## Getting Started

To get started, please familiarize yourself with the following:

*   **`config/`:** This directory contains the configuration files for the system.
*   **`core/`:** This directory contains the core components of the system.
*   **`docs/`:** This directory contains the documentation for the system.
*   **`tests/`:** This directory contains the tests for the system.

## Contribution Guidelines

Please follow these guidelines when contributing to the ADAM project:

*   Write clear and concise commit messages.
*   Update the documentation when adding new features or changing existing ones.
*   Write unit tests for all new code.
*   Ensure that all tests pass before submitting a pull request.

Thank you for your contributions to the ADAM project!
