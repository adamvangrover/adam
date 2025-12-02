# Roadmap: Agent System Expansion and Standardization

This roadmap outlines the steps to "add, update, and expand" the agent ecosystem within the ADAM project. The goal is to standardize all agents on the v22/v23 architecture (Async, AgentBase, Pydantic) and ensure comprehensive coverage as described in the Agent Catalog.

## Phase 1: Audit and Planning (Current)
- [x] Analyze `core/agents/` structure and `AGENT_CATALOG.md`.
- [x] Identify legacy agents (not inheriting `AgentBase`).
- [x] Identify missing agents (in catalog but not in file system, or vice versa).
- [ ] Create a "Manifest of Work" (this file).

## Phase 2: Standardization (High Priority)
Update core agents to inherit from `core.agents.agent_base.AgentBase` and implement the `execute` method.
- [ ] `RiskAssessmentAgent` (`core/agents/risk_assessment_agent.py`)
- [ ] `MarketSentimentAgent` (`core/agents/market_sentiment_agent.py`)
- [ ] `FundamentalAnalystAgent` (`core/agents/fundamental_analyst_agent.py`)
- [ ] `SNCAnalystAgent` (`core/agents/snc_analyst_agent.py`)
- [ ] `AlgoTradingAgent` (`core/agents/algo_trading_agent.py`)

**Standardization Requirements:**
*   Inherit from `AgentBase`.
*   `__init__` calls `super().__init__`.
*   Implement `async execute(self, **kwargs)`.
*   Use `logging` instead of `print`.
*   Type hinting (Pydantic preferred for inputs/outputs).

## Phase 3: Expansion (New Agents & Capabilities)
Create files for agents that exist in the catalog or concept but are missing or empty.
- [ ] `RedTeamAgent` (Verify and expand)
- [ ] `ReflectorAgent` (Verify and expand)
- [ ] `MetaCognitiveAgent` (Verify and expand)
- [ ] `BehavioralEconomicsAgent` (Verify and expand)
- [ ] `CrisisSimulationAgent` (Already looks good, check if accessible in `core/agents/` or needs link).

## Phase 4: Integration
- [ ] Ensure agents are correctly referenced in `config/agents.yaml` (if it exists).
- [ ] Update `AGENT_CATALOG.md` to reflect the actual codebase state (add new agents, fix paths).

## Phase 5: Verification
- [ ] Run basic import tests to ensure no syntax errors.
- [ ] Run `tests/test_agents.py` (if exists) or create a simple verification script.
