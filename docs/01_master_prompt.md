# FO SUPER-APP — MASTER SYSTEM PROMPT

“Unified Front Office app for markets, pricing, rating, execution, analytics, and autonomous strategy control.”

---

## 1. SYSTEM OVERVIEW

Build a Front Office Super-App that integrates:
*   Markets / Pricing Engine
*   IB + WM + AM intelligence layer
*   Credit underwriting + PD modeling + regulatory rating engine
*   Strategy Generator & Execution Router
*   Portfolio state + risk analytics
*   Local Sandbox Memory Store (Personal Knowledge Graph)
*   LLM Plugin Environment (MCP) for tools, agents, and skills

The platform should:
1.  Price competitively against banks and institutions
2.  Generate, rate, and execute strategies against real-time and historical data
3.  Unify Investment Banking (IB), Wealth Management (WM), and Asset Management (AM) views
4.  Provide a personal local-memory “co-pilot brain”
5.  Support LLM-driven multi-agent control via the Model Context Protocol (MCP)
6.  Enable easy UI for non-technical users with instant switching into terminal mode for power users

---

## 2. CORE GOALS

### 2.1 Market Functionality
*   Multi-asset live pricing
*   Auto-quoting & spreads
*   Competitive market making simulations
*   Automated backtesting & replay
*   Execution-quality benchmarking
*   Real-time risk metrics (VAR, Greeks, convexity, cross-exposures)

### 2.2 Strategy Layer
*   Alpha signal ingestion (structured + unstructured data)
*   LLM-assisted strategy drafting
*   RL agent for iterative optimization
*   Cross-asset hedging engine
*   Credit + equity + derivatives stack
*   PMC (Portfolio Management Console) running scenario frameworks (e.g., Market Mayhem)

### 2.3 Credit & Ratings Layer
*   S&P-like PD scoring engine
*   Regulatory rating assignment (Pass → Loss)
*   Probability distributions for default / migration
*   Credit memos generated on-demand
*   Sector modules (TMT & Healthcare priority)

### 2.4 Personal Sandbox Memory
*   Local vector store (e.g., sqlite + embeddings)
*   Secure offline knowledge retention
*   Plans, strategies, notes, preferences, workflows
*   Personal API that agents reference to maintain long-horizon consistency
*   Periodic sync with LLM through MCP memory tool

### 2.5 LLM / MCP Control Layer
*   Tools for:
    *   data query
    *   market pricing
    *   risk calculation
    *   order routing
    *   file synthesis
    *   strategy generation
    *   scenario engines
    *   MCP plugin definitions for each module
*   “Agent Stack”:
    *   Strategy Agent
    *   Risk Agent
    *   Market Agent
    *   Execution Agent
    *   Memory Agent
    *   Compliance Agent
    *   Repo Agent (Auto-file generator)

### 2.6 UI / UX
*   Clean web UI with:
    *   dashboard tiles (Markets, Credit, Strategies, Memory, Execution)
    *   one-click “Live Terminal Mode”
    *   guided walkthrough setup (for non-technical users)
    *   drag-and-drop data ingestion
    *   All features callable via API or conversational interface

---

## 3. REPOSITORY STRUCTURE (SKELETON)

```
/fo-superapp/
│
├─ /core/
│   ├─ pricing_engine/
│   ├─ credit_ratings/
│   ├─ market_data/
│   ├─ execution_router/
│   └─ risk_engine/
│
├─ /strategy/
│   ├─ alpha_signals/
│   ├─ rl_optimizer/
│   ├─ scenario_engine/
│   └─ cross_asset/
│
├─ /mcp/
│   ├─ tools/
│   ├─ agents/
│   └─ protocol_specs/
│
├─ /memory/
│   ├─ local_vector_store/
│   ├─ embeddings/
│   └─ sync_interface/
│
├─ /ui/
│   ├─ web/
│   └─ terminal/
│
├─ /api/
│   ├─ python/
│   └─ typescript/
│
├─ /docs/
│   ├─ architecture.md
│   ├─ roadmap.md
│   ├─ models.md
│   └─ workflows/
│
└─ README.md
```

---

## 4. MCP TOOL DEFINITION BLUEPRINT

Each module exposes:
*   invoke()
*   validate()
*   schema()

Example:

```json
{
  "name": "price_asset",
  "description": "Returns a competitive institutional-grade price for asset.",
  "input_schema": {
    "symbol": "string",
    "side": "buy | sell",
    "size": "float",
    "constraints": "object"
  },
  "output_schema": {
    "price": "float",
    "spread": "float",
    "confidence": "float"
  }
}
```

Do this for:
*   price_asset
*   generate_strategy
*   execute_order
*   rate_credit
*   assign_reg_rating
*   create_credit_memo
*   load_memory
*   write_memory
*   retrieve_market_data
*   run_backtest
*   generate_files

---

## 5. PERSONAL MEMORY ENGINE SPEC

### 5.1 Purpose

A dynamic personal knowledge graph storing:
*   user preferences
*   investment philosophy
*   past decisions
*   recurring playbooks
*   agent notes
*   research trails
*   proprietary signals

### 5.2 Data Storage
*   Sqlite + Chroma/FAISS vectors
*   Pydantic typed schemas
*   Buckets:
    *   “Personal Philosophy”
    *   “Playbooks”
    *   “Credit Framework”
    *   “Execution Logs”
    *   “Core Beliefs”
    *   “Agent Sync Notes”

### 5.3 Interaction
*   Agents use memory to personalize outputs
*   Memory Agent enforces consistency
*   User controls permissions and purge rules

---

## 6. SYSTEM MODES

### Mode 1 — Console Mode (“Trader Terminal”)
*   Full command-line execution
*   Scriptable strategies
*   Multi-agent orchestration

### Mode 2 — Guided Mode (“Wealth + Family Office Mode”)
*   Natural language
*   Recommendations + justifications
*   Easy portfolio actions

### Mode 3 — Developer Mode
*   Access tools directly
*   Inspect API
*   Generate files
*   Build custom agents

---

## 7. INIT PROMPT FOR LLM PLUGIN

Place this inside the MCP “core prompt”:

> You are the FO Super-App LLM controller.
> Your job is to direct tools, create strategies, rate credit, run executions,
> pull context from personal memory, and maintain a unified understanding
> of Adam’s risk preferences, investment constraints, and goals.
>
> Always respond with the optimal blend of
> IB + WM + AM perspectives.
>
> Always use tools when available.
> Always update personal memory through MCP when new durable info appears.
> Always optimize for institutional-grade quality.
>
> You orchestrate all modules.

---

## 8. DELIVERABLES THE REPO SHOULD AUTO-GENERATE
*   Strategy memos
*   Trading plans
*   Credit memos
*   Ratings files
*   Risk dashboards
*   Market summaries
*   Scenario simulation packets
*   Code scaffolding via Repo Agent

---

## 9. FUTURE EXTENSIONS
*   Live brokerage routing
*   Multi-agent reinforcement loop
*   Custom synthetic data generator
*   Portfolio governance engine
*   Compliance via LLM “audit” agent
