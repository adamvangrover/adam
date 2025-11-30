# Adam v23.0 Agent Expansion Roadmap

## Overview
This roadmap outlines the strategy for expanding the "Adaptive" (v23) capabilities of the Adam system, specifically focusing on the transition from static agents to cyclical, graph-based reasoning engines.

## 1. Core Graph Expansion (The "Brain")
The v23 architecture relies on `LangGraph` to implement "System 2" thinking (slow, deliberative, self-correcting).

- [x] **Adversarial Analysis**: `RedTeamGraph` (`core/v23_graph_engine/red_team_graph.py`)
    - *Status*: Implemented.
    - *Goal*: Simulates adversarial scenarios to stress-test portfolios.
- [x] **ESG Analysis**: `ESGGraph` (`core/v23_graph_engine/esg_graph.py`)
    - *Status*: Implemented.
    - *Goal*: Evaluates Environmental, Social, and Governance factors with greenwashing detection.
- [x] **Compliance**: `RegulatoryComplianceGraph` (`core/v23_graph_engine/regulatory_compliance_graph.py`)
    - *Status*: Implemented.
    - *Goal*: Checks against multi-jurisdictional regulations (Basel III, GDPR).
- [ ] **Crisis Simulation**: `CrisisSimulationGraph` (`core/v23_graph_engine/crisis_simulation_graph.py`)
    - *Status*: **Pending**.
    - *Goal*: A specialized graph for macro-economic stress testing (e.g., "What if rates hit 8%?").
    - *Action*: Create this graph, integrating logic from the existing `CrisisSimulationMetaAgent`.

## 2. Agent Wrapper Modernization (The "Body")
Legacy agents (v21) need to be updated to either:
1. Wrap a v23 Graph (acting as an API surface).
2. Inherit from the async `AgentBase` (v22).

- [ ] **CrisisSimulationMetaAgent**: `core/agents/meta_agents/crisis_simulation_agent.py`
    - *Action*: Update to use `CrisisSimulationGraph` if available, or enhance prompt-based logic.
- [ ] **RedTeamAgent**: `core/agents/red_team_agent.py`
    - *Action*: Update to wrap `RedTeamGraph`. Currently uses random selection.
- [ ] **ReflectorAgent**: `core/agents/reflector_agent.py`
    - *Action*: **Create/Update**. This agent is crucial for "Meta-Cognition" (analyzing the system's own thought process).
- [ ] **RiskAssessmentAgent**: `core/agents/risk_assessment_agent.py`
    - *Action*: Ensure it utilizes the `CyclicalReasoningGraph` or `SNCGraph`.

## 3. Orchestration Updates
The `MetaOrchestrator` must effectively route queries to these new specialized engines.

- [ ] **MetaOrchestrator**: `core/v23_graph_engine/meta_orchestrator.py`
    - *Action*: Add routing logic for "Crisis" or "Simulation" queries to the new `CrisisSimulationGraph`.

## 4. Documentation
- [ ] Update `AGENT_CATALOG.md` with new capabilities.
- [ ] Update `AGENTS.md` to reflect the v23 Graph patterns.

## Execution Strategy
Work will proceed in parallel tracks:
1. **Graph Track**: Building the `CrisisSimulationGraph`.
2. **Agent Track**: Updating wrappers and creating `ReflectorAgent`.
3. **Integration Track**: Wiring it all into `MetaOrchestrator`.
