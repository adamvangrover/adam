# Adam v23.5 "AI Partner" Architecture Guide

## Overview

The Adam v23.5 upgrade transforms the system into a **Hyper-Dimensional Knowledge Graph (HDKG)** generator. Unlike previous versions which focused on data retrieval or simple graph extraction, v23.5 acts as an autonomous financial analyst capable of "Deep Dive" due diligence.

## The "Deep Dive" Protocol

The core execution engine follows a strict 5-Phase sequential workflow:

### Phase 1: Entity & Management (The Foundation)
*   **Agent:** `ManagementAssessmentAgent`
*   **Function:** Resolves the legal entity structure and assesses management quality.
*   **Key Outputs:** `capital_allocation_score`, `key_person_risk`.

### Phase 2: Deep Fundamental & Valuation (The Equity Lens)
*   **Agents:** `FundamentalAnalystAgent`, `PeerComparisonAgent`
*   **Function:** Performs intrinsic valuation (DCF) and relative valuation (Multiples).
*   **Key Outputs:** `dcf_model`, `price_targets`.

### Phase 3: Credit, Covenants & SNC Ratings (The Debt Lens)
*   **Agents:** `SNCRatingAgent`, `CovenantAnalystAgent`
*   **Function:** Analyzes debt facilities, calculates covenant headroom, and assigns regulatory ratings.
*   **Key Outputs:** `regulatory_rating` (Pass/SpecialMention/Substandard), `covenant_headroom`.

### Phase 4: Risk, Simulation & Quantum Modeling (The Stress Test)
*   **Agents:** `MonteCarloRiskAgent`, `QuantumScenarioAgent`
*   **Function:** Simulates thousands of future paths for EBITDA and models "Black Swan" events.
*   **Key Outputs:** `monte_carlo_default_prob`, `quantum_scenarios`.

### Phase 5: Synthesis (The Verdict)
*   **Agent:** `PortfolioManagerAgent`
*   **Function:** Synthesizes all prior phases into a final conviction level.
*   **Key Outputs:** `conviction_level` (1-10), `recommendation`.

## Hyper-Dimensional Knowledge Graph (HDKG)

The output of the v23.5 pipeline is a strictly typed JSON object defined in `core/schemas/v23_5_schema.py`.

### Schema Structure
The `V23KnowledgeGraph` contains:
1.  **Meta:** Metadata about the target and generation time.
2.  **Nodes:**
    *   `entity_ecosystem`
    *   `equity_analysis`
    *   `credit_analysis`
    *   `simulation_engine`
    *   `strategic_synthesis`

## Orchestration

The `MetaOrchestrator` (`core/engine/meta_orchestrator.py`) handles the routing.
*   **Trigger:** If `complexity == "DEEP_DIVE"` (e.g., query contains "deep dive" or context flag is set).
*   **Flow:** Sequentially executes agents, passing state objects to build the HDKG.

## Agentic Handoff

Agents are designed to be modular.
*   `MonteCarloRiskAgent` relies on `ebitda` input which can come from `FundamentalAnalystAgent` (Phase 2).
*   `PortfolioManagerAgent` (Phase 5) requires inputs from all prior phases to form a verdict.
