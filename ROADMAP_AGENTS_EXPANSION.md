# Roadmap: Agent System Expansion and Standardization

This roadmap outlines the steps to "add, update, and expand" the agent ecosystem within the ADAM project. The goal is to standardize all agents on the v22/v23 architecture (Async, AgentBase, Pydantic) and ensure comprehensive coverage as described in the Agent Catalog.

## Phase 1: Audit and Planning (Completed)
- [x] Analyze `core/agents/` structure and `AGENT_CATALOG.md`.
- [x] Identify legacy agents (not inheriting `AgentBase`).
- [x] Identify missing agents (in catalog but not in file system, or vice versa).
- [x] Create a "Manifest of Work" (this file).

## Phase 2: Standardization (Completed)
Update core agents to inherit from `core.agents.agent_base.AgentBase` and implement the `execute` method.
- [x] `RiskAssessmentAgent` (`core/agents/risk_assessment_agent.py`)
- [x] `MarketSentimentAgent` (`core/agents/market_sentiment_agent.py`)
- [x] `FundamentalAnalystAgent` (`core/agents/fundamental_analyst_agent.py`)
- [x] `SNCAnalystAgent` (`core/agents/snc_analyst_agent.py`)
- [x] `AlgoTradingAgent` (`core/agents/algo_trading_agent.py`)

**Standardization Requirements:**
*   [x] Inherit from `AgentBase`.
*   [x] `__init__` calls `super().__init__`.
*   [x] Implement `async execute(self, **kwargs)`.
*   [x] Use `logging` instead of `print`.
*   [x] Type hinting (Pydantic preferred for inputs/outputs).

## Phase 3: Expansion (Completed)
Create files for agents that exist in the catalog or concept but are missing or empty.
- [x] `RedTeamAgent` (Verify and expand)
- [x] `ReflectorAgent` (Verify and expand)
- [x] `MetaCognitiveAgent` (Verify and expand)
- [x] `BehavioralEconomicsAgent` (Verify and expand)
- [x] `CrisisSimulationAgent` (Already looks good, check if accessible in `core/agents/` or needs link).

## Phase 4: Integration (Completed)
- [x] Ensure agents are correctly referenced in `config/agents.yaml` (if it exists).
- [x] Update `AGENT_CATALOG.md` to reflect the actual codebase state (add new agents, fix paths).

## Phase 5: Verification (Completed)
- [x] Run basic import tests to ensure no syntax errors.
- [x] Run `tests/verify_agents_refactor.py` or create a simple verification script.

# ROADMAP: ENTERPRISE FINANCIAL OPERATING SYSTEM UPGRADE

This master roadmap supersedes the previous agent expansion plan. It focuses on the Top 5 priority upgrades required to transition the ADAM repository into a fully autonomous, enterprise-grade system.

## Priority 1: The System 2 Upgrade (LangGraph State Machine)
Current predictive scripts run linearly. We must wrap reasoning agents in a cyclic state machine to introduce "Reflexion Loops" (self-correction).
- [x] Initialize `core/engine/state_machine.py` (LangGraph implementation).
- [x] Refactor `quantum_market_simulator.py` to function as an invocable graph node.
- [x] Create a `FinancialValidationReflector` node to validate DCF accounting constraints and force recalculation loops if violated.

## Priority 2: Active Memory Engine (Vector + Live Graph Database)
Transition away from reliant, static flat JSON files (`knowledge_graph.json`).
- [x] Implement `core/system/vector_memory.py` (e.g., ChromaDB integration) for semantic RAG of text nodes and historical intelligence.
- [x] Implement `core/system/graph_memory.py` (e.g., Neo4j/NetworkX) for executing dynamic relationship queries (e.g., Supply Chain linkages).
- [x] Upgrade the `Orchestrator` to natively query these dynamic databases to build context prior to agent invocation.

## Priority 3: System 1 Reality (Asynchronous WebSocket Ecosystem)
Implement the "Neural Swarm" of high-speed micro-workers dropping "Pheromones" on anomalies to automatically trigger the heavy System 2 LangGraphs.
- [x] Deploy an `asyncio` event loop and Redis Pub/Sub framework for continuous background processing.
- [x] Build light async micro-workers that subscribe to simulated market data.
- [x] Implement the `PheromoneEngine` to aggregate alert signals and trigger a System 2 deep-dive when thresholds are breached.t_stream_worker.py` to simulate WebSocket anomaly detection.

## Priority 4: The Consensus Arbitrator (Conflict Resolution)
Multiple agents will predictably disagree (e.g., Macro Bearish vs. Algo Bullish). The system must algorithmically judge these conflicts.
- [x] Create a `ConsensusEngine` module to arbitrate conflicting outputs based on historical accuracy and user-defined risk profiles.
- [x] Implement Human-In-The-Loop (HITL) deadlocks for uniformly high-confidence opposing viewpoints.rically opposed outputs.

## Priority 5: Eradicating Technical Debt (Agent Standardization)
Enforce the "Product" bifurcation rule. All core agents must inherit from `AgentBase` and use Pydantic validation to fail safely.
- [x] `core/agents/red_team_agent.py`
- [x] `core/agents/reflector_agent.py`
- [x] `core/agents/meta_cognitive_agent.py`
- [x] `core/agents/behavioral_economics_agent.py`
- [x] `core/agents/meta_agents/crisis_simulation_agent.py`
