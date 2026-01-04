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
*   [Agentic Oversight Framework (AOF)](#agentic-oversight-framework-aof)
*   [Getting Started](#getting-started)
*   [Contribution Guidelines](#contribution-guidelines)
*   [Operational Notes & Troubleshooting](#operational-notes--troubleshooting-v23-update)
*   [Directives for v25 Development](#directives-for-v25-development)

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

**Note on Directory Structure:**
The active implementation of the v23 Adaptive System is located in `core/engine/`.
The directory `core/v23_graph_engine/` contains legacy/proto-v23 artifacts and is maintained for backward compatibility.

Key Graph Engines:
*   **Red Team Graph:** Adversarial stress testing (`core/engine/red_team_graph.py`).
*   **Crisis Simulation Graph:** Macro-economic scenario modeling (`core/engine/crisis_simulation_graph.py`).
*   **ESG & Compliance Graphs:** Specialized domain reasoning.

### Swarm Architecture (v24 Preview)

Introduced in the "Strategic Technical Audit", the Swarm Architecture allows for massive parallelism via a Hive Mind.
*   **Hive Mind:** `core/engine/swarm/hive_mind.py`
*   **Pheromone Board:** `core/engine/swarm/pheromone_board.py`
*   **Workers:** `core/engine/swarm/worker_node.py`

The `MetaOrchestrator` is the unified entry point for all execution models, including v21 (synchronous), v22 (asynchronous), and v23 (graph-based). For the complete vision and technical specifications, please refer to the official `docs/v23_architecture_vision.md` mandate.

### Adam v23.5 "AI Partner" Upgrade

The v23.5 upgrade expands the system scope to a full-spectrum "Autonomous Financial Analyst", incorporating:
*   **Deep Credit:** SNC ratings and covenant analysis.
*   **Valuation:** DCF and multiples analysis.
*   **Risk:** Monte Carlo and Quantum scenarios.
*   **Strategic Synthesis:** M&A and conviction levels.

This upgrade is defined by the "Hyper-Dimensional Knowledge Graph" (HDKG) output schema and a new "Deep Dive" execution protocol. The system prompt is available in `config/Adam_v23.5_Portable_Config.json` and `prompt_library/Adam_v23.5_System_Prompt.md`.

## Agentic Oversight Framework (AOF)

As detailed in `docs/whitepapers/agentic_convergence_strategic_assessment.md`, the platform now implements a rigorous governance model for non-deterministic AI agents, managed by the "Head of AI Risk" persona.

*   **Automated Resolution Pathways (ARPs):** Rigid workflows that prevent agents from improvising in high-risk scenarios.
*   **Deterministic HITL Triggers:** Agents must halt and request human review if their internal conviction score falls below a set threshold (default 85%).
    *   Implemented via the `@AgenticOversightFramework.oversight_guardrail` decorator in `core/system/aof_guardrail.py`.
*   **Human-Machine Markdown (HMM) Protocol:** A structured communication standard (`core/system/hmm_protocol.py`) for analyst interventions and overrides.
    *   Requests use `HMM INTERVENTION REQUEST`.
    *   Logs use `HMM ACTION LOG`.
*   **"Four Eyes" Principle:** Critical outputs require dual verification.

## Getting Started

To get started, please familiarize yourself with the following:

*   **`config/`:** This directory contains the configuration files for the system.
*   **`core/`:** This directory contains the core components of the system.
    *   **`core/engine/`:** Contains the "Adaptive" system logic, implemented as LangGraph reasoning loops (e.g., `RedTeamGraph`, `CrisisSimulationGraph`).
*   **`docs/`:** This directory contains the documentation for the system.
*   **`tests/`:** This directory contains the tests for the system.

## Contribution Guidelines

Please follow these guidelines when contributing to the ADAM project:

*   Write clear and concise commit messages.
*   Update the documentation when adding new features or changing existing ones.
*   Write unit tests for all new code.
*   Ensure that all tests pass before submitting a pull request.

## Operational Notes & Troubleshooting (v23 Update)

*   **Entry Point:** The main entry point for the system is now `scripts/run_adam.py`. It uses the `MetaOrchestrator` to route queries to the appropriate engine (v23 Adaptive, v22 Async, or v21 Legacy).
*   **Dependencies:**
    *   `facebook-scraper` has been removed due to conflicts with `semantic-kernel`. Social media data sources will gracefully degrade if this package is missing.
    *   `langgraph` and `tiktoken` are required for v23 functionality.
*   **Logging:** A new `core/utils/logging_utils.py` module has been added to standardize logging configuration.
*   **Missing Files:** If you encounter `Knowledge base file not found` errors, ensure `data/risk_rating_mapping.json` exists. A default one is created if missing in some tests, but should be present in `data/` for production.
*   **Testing:** Always run tests using `python3 -m pytest` to ensure the correct virtual environment is used.

## Agent Prompts (v2.0)

The prompt library has been updated with v2.0 standards, emphasizing structured tool use (JSON Schema) and iterative workflows (Plan-Execute-Reflect).

*   **Canonical Agent Core (Recommended):** `prompt_library/system/agent_core.md` - The new reference implementation for all general-purpose agents. It defines the standard for role, behavior, and output requirements.
*   **Credit Risk Architect:** `prompt_library/v2_0_credit_risk_architect.md` - The flagship agent prompt using MCP standards.

## Alphabet Ecosystem Integration (v24.0 Preview)

The system is now integrating the full "Alphabet Ecosystem" (Gemini, Vertex AI, DeepMind Research) to create a Universal Financial Intelligence System.

*   **Gemini Financial Report Analyzer:** (`core/analysis/gemini_analyzer.py`) Performs deep qualitative analysis, extracting risk factors, sentiment, and ESG metrics using Gemini 1.5 Pro.
*   **Audio Financial Analyzer:** (`core/analysis/multimodal_analyst.py`) Analyzes earnings call audio for sentiment and speaker intent (Multimodal).
*   **RAG Financial Analyzer:** (`core/analysis/rag_analyzer.py`) Uses a lightweight RAG engine to synthesize insights from large document sets.
*   **AlphaAgent:** (`core/simulations/alpha_finance.py`) An RL agent trained in the `AlphaFinance` environment for portfolio optimization (DeepMind AlphaZero inspiration).

## Quantum-AI Convergence (New Capabilities)

Following the strategic analysis in `docs/whitepapers/quantum_ai_convergence.md`, the following modules have been added to the system:

*   **End-to-End Quantum Monte Carlo:** `core/v22_quantum_pipeline/qmc_engine.py` implements the Matsakos-Nield framework for simulating stochastic processes on quantum circuits.
*   **Generative Risk Engine:** `core/vertical_risk_agent/generative_risk.py` provides GAN-based market scenario generation for tail risk stress testing.
*   **Explainable Quantum AI:** `core/xai/iqnn_cs.py` implements the Interpretable Quantum Neural Network (IQNN-CS) framework with Inter-Class Attribution Alignment (ICAA) metrics.

## AI Partner v23.5 Upgrade ("The Omniscient Analyst")

The v23.5 upgrade massively expands the scope of the system from a simple graph extractor to a full-spectrum **Autonomous Financial Analyst**.

*   **New "Deep Dive" Pipeline:** A 5-phase execution protocol covering Entity Resolution, Deep Fundamental/Valuation, Credit/SNC Ratings, Risk/Quantum Simulation, and Strategic Synthesis.
*   **Omniscient State:** A new `OmniscientState` in `core/engine/states.py` supports the hyper-dimensional knowledge graph output.
*   **Portable Prompt:** A comprehensive system prompt is available in `prompt_library/AOPL-v1.0/system_architecture/autonomous_financial_analyst_v23_5.md`.

## Directives for v25 Development

### 1. The Prime Directive: Bifurcation
This repository implements two distinct strategies. **Do not mix them.**
- **Path A (`core/vertical_risk_agent`)**: Prioritize **Reliability**, **Auditability**, and **Business Logic**. Code should be defensive, heavily typed (Pydantic), and explained via logs. Performance is secondary to correctness.
- **Path B (`experimental/inference_lab`)**: Prioritize **Velocity**, **Throughput**, and **Math**. Code should be optimized (Triton/CUDA), minimal, and benchmarked. Business logic is irrelevant here.

### 2. Coding Standards

#### Path A (Product)
- **Imports**: Absolute imports preferred.
- **Error Handling**: Never crash. Use `try/except` blocks with logging.
- **State**: All agent state must be defined in `state.py` using `TypedDict` or `Pydantic`.
- **Tools**: All external interactions (API, DB) must go through the MCP Server pattern (`tools/mcp_server`).

#### Path B (Research)
- **Imports**: Minimal overhead.
- **Memory**: Be conscious of VRAM. Use `inplace` operations where possible.
- **Comments**: Explain the *math* behind the optimization (e.g., "We use a block size of 128 to align with warp size").

### 3. Workflow Protocol
1. **Plan**: Before writing code, inspect the `README.md` of the target directory.
2. **Visualize**: If modifying `langgraph` flows, generate the Mermaid diagram to verify logic.
3. **Verify**:
   - Path A: Run `python evals/run.py`.
   - Path B: Run `python benchmarks/benchmark_tps.py`.

### 4. Known Context
- The system uses a mock `langgraph` if the library is missing. Do not remove this fallback unless you are installing the real dependency.
- The `xbrl_handler.py` has a real XML parser but falls back to mock data if the file is missing. This is intentional for demo purposes.

### 5. Documentation
- Update `docs/v25_strategic_divergence_roadmap.md` when you complete a major feature.
- Keep `outstanding_errors.md` updated if you encounter intractable issues.

Thank you for your contributions to the ADAM project!
