# Risk Consensus Framework

## Overview

The Risk Consensus Framework is the implementation of the "Bicameral Risk Mind" concept described in the Agentic Convergence whitepaper. It acknowledges that "Risk" is not a single objective truth but a negotiation between **Regulatory Compliance** (The Rules) and **Economic Reality** (The Math).

## Architecture

The framework utilizes two distinct agents for every major risk assessment (like SNC Ratings):

1.  **Regulatory Agent (The "brake"):**
    *   **Persona:** Government Examiner / Regulator.
    *   **Logic:** Deterministic, rigid, rule-based.
    *   **Source:** Interagency Guidance, Basel III.
    *   **Goal:** Ensure we do not violate the law.

2.  **Strategic Agent (The "gas"):**
    *   **Persona:** Risk Officer / Portfolio Manager.
    *   **Logic:** Probabilistic, forward-looking, cash-flow based.
    *   **Source:** Market data, DSCR, Monte Carlo.
    *   **Goal:** Identify economic value and hidden risks.

## The Consensus Engine

The `RiskConsensusEngine` (`core/engine/risk_consensus_engine.py`) synthesizes these two inputs into a final `SystemConviction`.

### Mathematical Logic

$$
\mathbb{C}(x) = \alpha \cdot \mathbb{I}(M_{reg} = M_{strat}) + \beta \cdot \text{conf}(M_{strat}) - \gamma \cdot \text{div}(M_{reg}, M_{strat})
$$

### Decision States

| Regulatory Rating | Strategic Rating | State | Action |
| :--- | :--- | :--- | :--- |
| **Pass** | **Pass** | **Consensus** | High Conviction. Automated Approval. |
| **Fail** | **Pass** | **Regulatory Constraint** | The deal is good, but rules forbid it. *Structure Required.* |
| **Pass** | **Fail** | **Hidden Risk** | The deal looks compliant but is rotton. *Hard Reject.* |
