# Agent Capabilities: Adam v23.5 Upgrade

This document outlines the expanded capabilities of the Adam agent swarm following the v23.5 "Sovereign Financial Intelligence" upgrade.

## 1. New Specialized Agents

### RedTeamAgent (Adversarial Adversary)
*   **Role**: Internal Auditor & Stress Tester.
*   **Architecture**: Implements a self-contained **Cyclical Reasoning Loop** using LangGraph.
*   **Core Skill**: `CounterfactualReasoningSkill`.
*   **Workflow**:
    1.  **Generate**: Creates "Bear Case" scenarios by inverting key assumptions in investment memos (e.g., flipping "Growth" to "Contraction").
    2.  **Simulate**: Estimates the financial impact (VaR/CVaR) of the scenario.
    3.  **Refine**: Automatically escalates the severity of the scenario if the initial impact is too mild, ensuring robust stress testing.

## 2. Enhanced Financial Engines

### Quantum Risk Engine (QAE)
*   **Goal**: Model "Black Swan" events and tail risks more accurately than classical methods.
*   **Implementation**: `core/risk_engine/quantum_model.py`.
*   **Dual-Path Execution**:
    *   **Quantum Path**: Uses Qiskit's `IterativeAmplitudeEstimation` (IQAE) to achieve quadratic speedup in VaR estimation (Simulated via Aer Backend).
    *   **Classical Path**: Falls back to a Numpy-based Geometric Brownian Motion (GBM) Monte Carlo simulation (10,000 paths) if quantum resources are unavailable.

### MCP Server (Model Context Protocol)
*   **Goal**: Standardized tool exposure for the agent swarm.
*   **Tools**:
    *   `calculate_wacc`: Weighted Average Cost of Capital.
    *   `calculate_dcf`: Discounted Cash Flow.
    *   `calculate_quantum_risk`: Quantum-enhanced probability of default.
    *   `analyze_financial_sentiment`: NLP-based sentiment analysis.
*   **Resources**:
    *   `market_data://{ticker}`: Provides access to market baseline data (from `data/adam_market_baseline.json`).

## 3. Cognitive Core (HNASP)

### Hybrid Neurosymbolic Agent State Protocol
*   **Goal**: Provide structured, verifiable memory and logic for agents.
*   **Components**:
    *   **MetaNamespace**: Tracks security context and trace IDs.
    *   **LogicLayer**: Stores deterministic business rules (ASTs) executed via `json-logic`.
    *   **PersonaState**: Tracks agent personality using EPA (Evaluation, Potency, Activity) vectors.
    *   **ContextStream**: A structured log of all agent-user turns.
*   **Benefit**: Allows for "Time-Travel Debugging" and strict auditability of agent decisions.

## 4. Governance (EACI)

### Enterprise Adaptive Core Interface
*   **Goal**: Security and Compliance.
*   **Middleware**: Intercepts all prompts to sanitize input (prevent injection attacks) and inject Role-Based Access Control (RBAC) contexts.
*   **PromptOps**: Automated regression testing of prompts against a "Golden Dataset" to prevent drift.
