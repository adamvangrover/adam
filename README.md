# Adam v26.0: The Neuro-Symbolic Financial Sovereign

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![Docker Image](https://img.shields.io/badge/docker-ready-blue)](https://hub.docker.com/) [![FinanceBench](https://img.shields.io/badge/FinanceBench-99%25-green)](https://arxiv.org/abs/2311.11944)

> **Adam v26.0 upgrades financial AI from a conversational chatbot to a fiduciary architect. By fusing the intuitive speed of a Neural Swarm (System 1) with the deliberate logic of a Neuro-Symbolic Graph (System 2), Adam delivers calculated conviction for high-stakes capital allocation.**

---

## 📚 Quick Links

*   [**🚀 Launch Neural Dashboard**](showcase/index.html)
*   [**⚡ Setup Guide**](docs/setup_guide.md)
*   [**🤖 Agent Developer Bible**](AGENTS.md)
*   [**🧠 Agent Knowledge Base**](docs/AGENTS_KNOWLEDGE_BASE.md)
*   [**📖 Architecture Overview**](docs/architecture.md)
*   [**🎓 Tutorials**](docs/tutorials.md)
*   [**🎓 Office Nexus Tutorial**](docs/TUTORIAL_OFFICE_NEXUS.md)
*   [**📦 Custom Builds**](docs/custom_builds.md)

---

## 🧠 Why Adam? The "System 2" Revolution

The era of the "LLM Wrapper" is over. Institutional finance faces an **Epistemological Crisis**: stochastic models hallucinate, making them dangerous for due diligence.

**Adam v26.0** solves this by implementing a **Hybrid Cognitive Engine**:

### System 1: The Swarm (Fast)
*   **Role:** The Body. Handles perception, data ingestion, and news monitoring.
*   **Architecture:** Asynchronous Event Loop (Pub/Sub).
*   **Use Case:** "Monitor AAPL for breaking news."

### System 2: The Graph (Slow)
*   **Role:** The Brain. Handles reasoning, planning, and criticism.
*   **Architecture:** Directed Acyclic Graph (DAG) with feedback loops.
*   **Use Case:** "Write a 30-page investment memo on the solvency of distressed debt."

**Result:** Adam "thinks before he speaks," drafting, critiquing, and refining analysis before presenting it to the user.

### System Architecture

```mermaid
graph TD
    UserNode["User Analyst"] -->|HTTP| WebApp["Web App (React Flask)"]
    WebApp -->|MCP| MCPServer["MCP Server (Python)"]
    MCPServer -->|Router| MetaOrchestrator["Meta Orchestrator"]
    MetaOrchestrator -->|Fast Path| NeuralSwarm["System 1: Async Swarm"]
    MetaOrchestrator -->|Slow Path| NeuroPlanner["System 2: Neuro-Symbolic Planner"]
    NeuroPlanner -->|Task| FundAgent["Fundamental Agent"]
    NeuroPlanner -->|Task| RiskAgent["Risk Agent"]
    FundAgent -->|Data| CredSentinel["Credit Sentinel"]
    RiskAgent -->|Data| CredSentinel
```

---

## 📂 Directory Structure

A high-level overview of the repository layout:

```text
adam/
├── core/                   # The "Brain" (See core/README.md)
│   ├── agents/             # Specialized autonomous agents
│   ├── engine/             # Neuro-Symbolic Planner & Orchestrator
│   ├── credit_sentinel/    # Distressed Debt Analysis Module
│   └── system/             # "System 1" Async Swarm infrastructure
├── services/
│   └── webapp/             # React/Flask "Neural Dashboard"
├── showcase/               # Static HTML visualizers and demos
├── docs/                   # Documentation, tutorials, and guides
├── scripts/                # Utility scripts for running and testing
├── prompt_library/         # The "Mind" (AOPL v26.0 Prompts)
└── server/                 # MCP Server implementation
```

---

## ⚡ Getting Started

We strictly use **`uv`** for lightning-fast, reproducible Python environment management.

### Prerequisites

*   **OS:** Linux, macOS, or Windows (WSL2 recommended)
*   **Tooling:** `uv` (Modern Python Package Manager)
*   **API Keys:** OpenAI (GPT-4), Anthropic (Claude 3.5), or local LLM.

### Quick Start

1.  **Install `uv` (if not installed):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/adamvangrover/adam.git
    cd adam
    ```

3.  **Sync Dependencies:**
    ```bash
    # This installs Python and all dependencies in seconds
    uv sync
    ```

4.  **Activate Environment:**
    ```bash
    source .venv/bin/activate
    ```

5.  **Launch the System:**
    ```bash
    # Run the interactive CLI
    python scripts/run_adam.py
    ```

For detailed instructions, see the [Setup Guide](docs/setup_guide.md).

---

## 🏰 Platform Capabilities

Adam unifies three critical domains into a single cognitive architecture:

### 1. Distressed Debt & Credit (Credit Sentinel)
*   **ICAT Engine:** Python-based 3-statement modeling and DCF valuation.
*   **SNC Rating:** Automating regulatory grading (Pass vs. Substandard).
*   **Covenant Analysis:** Extracting and stress-testing debt covenants.

### 2. Quantitative Engineering
*   **Risk Modeling:** Deterministic calculation of VaR, Sharpe, and Sortino ratios.
*   **Factor Analysis:** Regression against Fama-French factors.

### 3. Agentic Workflow
*   **Meta-Orchestration:** Dynamic routing of queries to specialized agents.
*   **Consensus Engine:** Aggregating multi-agent perspectives into a single conviction score.
*   **Governance Layer:** API Gatekeeper and Structured Narrative Logging for enterprise safety.

---

## 📦 Custom Builds & Export

Adam includes a powerful build system to create self-contained, portable environments.

### Interactive Builder

Run the build wizard to create a custom distribution with specific modules, runtime profiles, and Docker support:

```bash
python scripts/build_adam.py
```

See [**Custom Builds Documentation**](docs/custom_builds.md) for details.

### Quick Export

To export a single module directly:

```bash
python scripts/export_module.py market_mayhem --output my_exports
```

---

## 🗺️ Roadmap: Path to Autonomy

*   **Phase 1 (Current): The Autonomous Analyst.** Deep Dives, Credit Memos, and Regulatory Grading.
*   **Phase 1.5 (ADAM-V-NEXT): The Command Center.** Synthesizer Dashboard, Agent Intercom, and War Room.
*   **Phase 2 (Q3 2025): The Portfolio Manager.** Multi-entity risk aggregation and automated rebalancing.
*   **Phase 3 (Q3 2026): The Market Maker.** High-frequency sentiment trading and liquidity provision via Quantum RL.

---

## 🤝 Contributing

We are building the open-source standard for institutional AI.

*   **Directives:** Please read [AGENTS.md](AGENTS.md) and the [Agent Knowledge Base](docs/AGENTS_KNOWLEDGE_BASE.md) before writing a single line of code.
*   **Process:** Read [CONTRIBUTING.md](CONTRIBUTING.md) for pull request guidelines.

### License

Distributed under the MIT License. See `LICENSE` for more information.

---

**For AI Agents and LLMs,** please see [llms.txt](llms.txt) (optimized) or [llms-full.txt](llms-full.txt) (comprehensive) for context.*
