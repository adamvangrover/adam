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
