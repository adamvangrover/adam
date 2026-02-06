# Cognitive Finance Research

This directory contains the theoretical foundations and whitepapers that drive the architecture of Adam v26.0.

## 📄 Key Papers & Concepts

### 1. Cognitive Finance Architecture (`cognitive_finance_architecture.md`)
**The Blueprint.**
Defines the transition from "Stochastic Parrots" (LLMs) to "Neuro-Symbolic Sovereigns".
*   **Key Concept:** The "System 1 vs. System 2" split.
    *   *System 1:* Fast, intuitive, pattern matching (Swarm).
    *   *System 2:* Slow, deliberate, logical reasoning (Graph Engine).

### 2. One-Shot World Models (`one_shot_world_models.md`)
**The Simulator.**
Describes how Adam builds an internal simulation of the market economy to test hypotheses ("Counterfactual Reasoning").
*   **Implementation:** See `core/simulations/world_model.py`.

### 3. Federated Learning (`federated_learning.md`)
**The Privacy Layer.**
Explains how Adam learns from distributed data (e.g., across different bank silos) without data ever leaving the premise.
*   **Implementation:** See `core/system/learning/trace_collector.py`.

### 4. Graph Neural Networks (`graph_neural_networks.md`)
**The Connector.**
Details the use of GNNs to model complex inter-dependencies between assets (e.g., Supplier-Customer relationships).
*   **Implementation:** See `core/v23_graph_engine/`.

## 📚 How to Use This Research

These documents are not just theory; they are **Specifications**.
*   Developers should read `cognitive_finance_architecture.md` before touching `core/engine/`.
*   Data Scientists should read `one_shot_world_models.md` before building new simulation scenarios.
