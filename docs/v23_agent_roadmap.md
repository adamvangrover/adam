# Adam v23.0 Agent Expansion & Migration Roadmap

This document outlines the strategic roadmap for migrating legacy v21 agents to the v23 "Adaptive System" architecture and expanding the agent ecosystem.

## Strategic Goals

1.  **Hybrid Architecture**: Leverage v22 Asynchronous messaging for execution and v23 Cyclical Graphs for reasoning.
2.  **Graph Wrapping**: Encapsulate sophisticated `LangGraph` logic within standard `AgentBase` wrappers to maintain a unified API.
3.  **Meta-Cognition**: Implement "System 2" thinking (critique, reflection, refinement) across all major analytical agents.

## Implementation Roadmap

### Phase 1: Core Graph Integration (Current Focus)

- [ ] **CrisisSimulationMetaAgent** (`core/agents/meta_agents/crisis_simulation_agent.py`)
    - **Objective**: Integrate `CrisisSimulationGraph` for dynamic scenario modeling.
    - **Action**: Update agent to wrap `core/engine/crisis_simulation_graph.py`.
    - **Fallback**: Maintain prompt-based legacy logic if dependencies fail.

- [ ] **RiskAssessmentAgent** (`core/agents/risk_assessment_agent.py`)
    - **Objective**: Adopt `CyclicalReasoningGraph` for deep credit risk analysis.
    - **Action**: Refactor to use `core/engine/cyclical_reasoning_graph.py`.
    - **Logic**: Use graph for deep dives, legacy rules for quick checks.

- [ ] **RedTeamAgent** (`core/agents/red_team_agent.py`)
    - **Objective**: Standardization.
    - **Status**: Already updated to wrap `RedTeamGraph`. Verified.

### Phase 2: Meta-Cognitive Expansion

- [ ] **ReflectorAgent** (`core/agents/reflector_agent.py`)
    - **Objective**: Move from heuristics to graph-based self-correction.
    - **Action**:
        1. Create `core/engine/reflector_graph.py` (Critique -> Refine Loop).
        2. Update agent to wrap this graph.

### Phase 3: New Agent Development

- [ ] **RegulatoryComplianceAgent** (`core/agents/regulatory_compliance_agent.py`)
    - **Objective**: Integrate `RegulatoryComplianceGraph`.
    - **Action**: Update wrapper to use the new graph engine.

- [ ] **ESGAgent** (`core/agents/industry_specialists/esg_agent.py` - if exists, else create)
    - **Objective**: Integrate `ESGGraph`.
    - **Action**: Ensure a wrapper exists for the ESG graph.

## Parallel Execution Tracks

| Track | Owner | Focus |
| :--- | :--- | :--- |
| **Graph Track** | Core Eng | Building `LangGraph` definitions in `core/engine/`. |
| **Wrapper Track** | Agent Eng | Updating `core/agents/` classes to import and invoke graphs. |
| **Orchestration** | System Arch | Updating `MetaOrchestrator` to route to new agents. |

## Technical Guidelines

- **Imports**: Always use `try/except` when importing from `core.engine` to ensure the system runs in "Mock Mode" if `langgraph` is missing.
- **State**: Use Pydantic models or `TypedDict` from `core.engine.states`.
- **Async**: All v23 integrations must use `await app.ainvoke(...)`.
