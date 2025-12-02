# Adam v23.5 Architecture Guide: The "AI Partner" Ecosystem

## Executive Summary

Adam v23.5 represents the evolution from an "Adaptive System" to an "Autonomous Financial Partner". While v23.0 introduced graph-based reasoning, v23.5 implements a rigid, rigorous **"Deep Dive" Protocol** designed to mimic the workflow of a senior institutional analyst.

The core directive is: **"Do not just retrieve data. Synthesize conviction."**

---

## 1. The Hyper-Dimensional Knowledge Graph (HDKG)

The output of v23.5 is not a text report, but a structured JSON object defined in `core/schemas/v23_5_schema.py`. This schema models the target entity across five dimensions:

1.  **Entity Ecosystem:** Legal structure, management assessment, and competitive moat.
2.  **Equity Analysis:** Deep fundamentals, DCF valuation, and peer multiples.
3.  **Credit Analysis:** Shared National Credit (SNC) ratings and covenant headroom.
4.  **Simulation Engine:** Monte Carlo solvency tests and Quantum "Black Swan" scenarios.
5.  **Strategic Synthesis:** Final conviction scoring (1-10) and buy/sell/hold verdict.

### Schema Location
- **Definition:** `core/schemas/v23_5_schema.py`
- **Output Artifact:** `V23KnowledgeGraph` (Pydantic Model)

---

## 2. The "Deep Dive" Execution Protocol

The `MetaOrchestrator` (`core/v23_graph_engine/meta_orchestrator.py`) routes queries with high complexity (e.g., "Deep dive on Apple") to the **Deep Dive Protocol**. This protocol executes in 5 sequential phases, orchestrating specialized agents.

### Phase 1: Entity & Management
- **Agent:** `ManagementAssessmentAgent`
- **Task:** Assess capital allocation (buybacks vs. capex), insider alignment, and CEO tone using NLP.
- **Output:** `EntityEcosystem` node.

### Phase 2: Deep Fundamental & Valuation
- **Agents:** `FundamentalAnalystAgent`, `PeerComparisonAgent`
- **Task:** Build a 2-stage DCF model and compare EV/EBITDA multiples against a dynamic peer set.
- **Output:** `EquityAnalysis` node.

### Phase 3: Credit, Covenants & SNC Ratings
- **Agents:** `SNCRatingAgent`, `CovenantAnalystAgent`
- **Task:** Mimic a regulatory exam. Classify debt as Pass/Special Mention/Substandard based on Leverage and Collateral Coverage. Check covenant headroom.
- **Output:** `CreditAnalysis` node.

### Phase 4: Risk, Simulation & Quantum Modeling
- **Agents:** `MonteCarloRiskAgent`, `QuantumScenarioAgent`
- **Task:** Run 10,000 Monte Carlo paths on EBITDA volatility to estimate Probability of Default (PD). Hallucinate "Black Swan" events (Quantum Scenarios).
- **Output:** `SimulationEngine` node.

### Phase 5: Synthesis & Conviction
- **Agent:** `PortfolioManagerAgent`
- **Task:** Weigh the outputs of Phases 1-4. If Equity is Bullish but Credit is Substandard, resolve the conflict. Assign a Conviction Score (1-10).
- **Output:** `StrategicSynthesis` node.

---

## 3. Implementation Details

### Specialized Agents (`core/agents/specialized/`)
All new agents inherit from `AgentBase` and implement strict Pydantic typing for inputs and outputs.
- `snc_rating_agent.py`: Implements OCC/Fed regulatory logic.
- `monte_carlo_risk_agent.py`: Implements NumPy-based stochastic modeling.

### Orchestration
The `MetaOrchestrator` attempts to run the `DeepDiveGraph` (LangGraph implementation). If the graph is unavailable or fails, it falls back to a robust **Manual Orchestration** method (`_run_deep_dive_manual_fallback`) that sequentially calls the specialized agents and assembles the HDKG.

---

## 4. Usage

To trigger the v23.5 pipeline, submit a query containing keywords like "deep dive", "valuation", or "full analysis".

```python
# Example Usage
from core.v23_graph_engine.meta_orchestrator import MetaOrchestrator

orchestrator = MetaOrchestrator()
result = await orchestrator.route_request("Perform a deep dive analysis on Tesla")
print(result)
```

## 5. Artifacts

- **Gold Standard Data:** `data/gold_standard/v23_5_knowledge_graph.json` contains comprehensive mock profiles for Apple, Ford, and GameStop, serving as the ground truth for UI development and testing.

---

**Built for Depth. Built for Conviction.**
*Adam v23.5 System Architect*
