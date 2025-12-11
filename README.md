# FO Super-App: The Unified Front Office

> **Vision:** A "Super-App" that unifies markets, ratings, execution, analytics, and personal memory into one autonomous architecture.

**FO Super-App** builds upon the Adam v23 foundation to create a complete institutional-grade platform. It integrates:
*   **Markets & Pricing:** Competitive market making and execution.
*   **Credit & Ratings:** S&P-like scoring and regulatory compliance.
*   **Strategy:** Alpha signal ingestion and RL-based optimization.
*   **Personal Memory:** A local "co-pilot brain" that learns your investment philosophy.
*   **MCP Control Layer:** Universal tool access for LLM agents.

## 🏰 FO Super-App: Family Office Edition

The system now includes a specialized "Family Office" layer unifying:
*   **Wealth Management:** Goal planning, Trust modeling.
*   **Investment Banking:** Deal flow screening and Deep Dive analysis.
*   **Asset Management:** Risk aggregation across multiple entities.
*   **Governance:** Automated Investment Policy Statement (IPS) generation.

[📚 Master Prompt](./docs/01_master_prompt.md) | [🏗️ System Architecture](./specs/system_architecture.md)

---

# Adam v23.5: Your AI-Powered Partner

> **Note:** This document describes the current active version of the Adam system (v23.0). For details on the legacy stable version, please see the v21.0 Documentation.

---

*   **Cyclical Reasoning Graph:** A self-correcting neuro-symbolic engine.
*   **Neural Dashboard:** Real-time visualization of agent thought processes.
*   **Hybrid Architecture:** Combining v21's reliability with v22's speed and v23's intelligence.
*   **Gold Standard Data Pipeline:** A rigorous "universal ingestion" process that scrubs and certifies all system knowledge.

## 🧠 Adam v23.0: The Adaptive Hive Mind

**Mission: Autonomous Financial Analysis & Adaptive Reasoning**

Adam has evolved. v23.0 introduces the **"Adaptive System" architecture**—a self-correcting, neuro-symbolic engine designed to perform deep financial deep dives, risk assessments, and market simulations with human-like reasoning and machine-speed execution. Unlike traditional chatbots, Adam "thinks" in graphs, critiquing its own work before presenting it to you.

[🚀 Launch Neural Dashboard](#) | [📖 Read the User Guide](#) | [⚡ Quick Start](#)

---

## 🚀 Mission Control
[**Launch Neural Dashboard**](./showcase/neural_dashboard.html)
Monitor real-time agent reasoning, knowledge graph updates, and risk simulations.

### 🏗️ System Architecture

Adam v23.0 moves beyond linear chains to a dynamic, graph-based execution model. The system creates a **"Cyclical Reasoning Graph"** for every query, allowing it to draft, critique, and refine its own analysis before presenting results.

```mermaid
graph TD
    User[User / API] -->|Query| Meta[Meta Orchestrator]
    
    subgraph "The Brain (v23 Graph Engine)"
        Meta --> Planner[Neuro-Symbolic Planner]
        Planner -->|Generates Path| Graph[Dynamic Reasoning Graph]
        
        Graph --> Node1[Data Retrieval]
        Graph --> Node2[Analysis Agent]
        Graph --> Node3[Risk Simulation]
        
        Node1 -->|Evidence| Critic[Self-Correction Loop]
        Node2 -->|Draft| Critic
        Node3 -->|Scenarios| Critic
        
        Critic -->|Refinement Needed| Graph
        Critic -->|Approved| Synthesis[Final Synthesis]
    end
    
    subgraph "Memory & Knowledge"
        KG[(Unified Knowledge Graph)]
        Vec[(Vector Store)]
    end
    
    Node1 <--> KG
    Node1 <--> Vec
    
    Synthesis -->|Final Report| Output[JSON / HTML / PDF]
```

#### Core Components

  * **Meta Orchestrator** (`core/engine/meta_orchestrator.py`): The central "cortex" that routes tasks, manages state, and orchestrates the swarm of specialized agents.
  * **Neuro-Symbolic Planner** (`core/engine/neuro_symbolic_planner.py`): Combines the creativity of LLMs with the logical rigor of Knowledge Graphs to plan execution paths that are both novel and grounded in fact.
  * **Cyclical Reasoning Engine** (`core/engine/cyclical_reasoning_graph.py`): A feedback loop (Draft -\> Critique -\> Refine) that ensures high conviction. It detects logical fallacies or missing data and automatically schedules remedial tasks.

-----

### 📊 Data & The Gold Standard Pipeline

Garbage in, garbage out. Adam v23.0 utilizes a rigorous **Gold Standard Data Pipeline** ("The Universal Ingestor") to ensure all insights are based on verified, high-quality data.

  * **Ingestion Sources:** Financial news APIs, SEC filings (XBRL), market data feeds (Bloomberg/AlphaVantage connectors), and government statistics.
  * **Scrubbing & Validation:** Every data point is scored for "Conviction" (0-100%) based on source reliability and cross-verification against the Knowledge Graph.
  * **Unified Format:** Data is normalized into a standard JSONL format for agent consumption.

[View Data Pipeline Documentation](https://www.google.com/search?q=%23)

-----

### 💡 Example Outputs

### 3. Gold Standard Data Pipeline
A new "Universal Ingestor" ensures that every piece of data in the system is high-quality.

*   **Ingest & Scrub:** Recursively scans reports, prompts, code, and data.
*   **Conviction Scoring:** Automatically assesses the quality and "conviction" of data (0-100%).
*   **Unified Access:** All data is normalized into a standard JSONL format accessible by any agent.
*   [Read the Pipeline Documentation](./docs/GOLD_STANDARD_PIPELINE.md)
Adam doesn't just chat; it produces structured, professional-grade financial artifacts ready for investment committees.

#### 1\. Strategic Deep Dive (JSON Snippet)

*Generated by the Omniscient Analyst (v23.5) - Full Template*

```json
{
  "report_id": "RPT-NVDA-2025-03",
  "entity": "NVIDIA Corp",
  "conviction_score": 94.5,
  "strategic_synthesis": {
    "outlook": "Bullish",
    "key_driver": "Sovereign AI adoption and B200 backlog saturation.",
    "risks": ["Supply chain concentration (TSMC)", "Geopolitical export controls"]
  },
  "valuation_models": {
    "dcf_implied_price": 1450.00,
    "peer_multiple_target": 1380.00,
    "sensitivity_analysis": "High sensitivity to datacenter capex reduction."
  },
  "generated_at": "2025-03-15T14:30:00Z"
}
```

#### 2\. Risk Assessment Matrix

*Adam automatically generates risk matrices for portfolio stress testing using the Monte Carlo Risk Agent.*

| Risk Category | Probability | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Market** | Medium | High | Hedge via inverse ETFs on semiconductor indices (SOXS). |
| **Credit** | Low | High | Monitor debt-to-equity ratios quarterly; currently stable at 0.8x. |
| **Geopolitical** | High | Severe | Diversify supply chain exposure outside of APAC region; monitor Taiwan Strait tensions. |
| **Regulatory** | Medium | Medium | Track DOJ antitrust probes into AI hardware bundling. |

-----

3.  **Run Adam:**
    ```bash
    python scripts/run_adam.py
    ```

4.  **View the Showcase:**
    Open `showcase/index.html` in your browser.

## 💰 Financial Engineering Platform (v23.5)

A modular, portable, and configurable Financial Engine for DCF Valuation, VC/LBO Sponsor Modeling, and Regulatory Credit Risk Analysis (SNC/Rating).

### 🚀 Launch Dashboard
[**Launch Financial Engine (Client Side)**](./showcase/financial_engineering.html)
Interactive dashboard for valuation, credit ratings, and sensitivity analysis.

### 🐍 Python Core
The core logic is available in `src/` and can be run as a Streamlit app.

```bash
# Run the Interactive Streamlit App
streamlit run app.py
```

### Modules
- **`src/core_valuation.py`**: Discounted Cash Flow (DCF), WACC, and Terminal Value logic.
- **`src/credit_risk.py`**: Credit Sponsor Model, Downside Sensitivity, and Regulatory Ratings (SNC).
- **`src/config.py`**: Global financial assumptions (Tax rates, Risk-free rates).

## 📂 Repository Structure

*   `core/engine/`: The heart of the new system.
    *   `cyclical_reasoning_graph.py`: The self-correcting analysis loop.
    *   `neuro_symbolic_planner.py`: The logic for pathfinding in the KG.
    *   `meta_orchestrator.py`: The central brain routing tasks.
*   `core/data_processing/`: Data ingestion and standardisation.
    *   `universal_ingestor.py`: The Gold Standard Pipeline.
*   `showcase/`: The "Mission Control" UI assets.
*   `data/`: Knowledge base and artisanal training sets.
*   `docs/`: Comprehensive documentation.

### 🚀 Key Capabilities

  * **Cyclical Reasoning:** Unlike standard chatbots, Adam iterates. If data is missing, it creates a sub-task to find it. If logic is flawed, it self-corrects.
  * **Quantum Risk Modeling:** (v23.5) Uses simulated quantum annealing (via `core/v22_quantum_pipeline/`) to model "Black Swan" events and their impact on complex portfolios.
  * **Traceability:** Every conclusion is back-linked to source documents in the Knowledge Graph (PROV-O ontology compliant).
  * **Multi-Modal Output:** Generates interactive HTML dashboards (`showcase/`), PDF reports, and raw JSON data streams.
  * **Specialized Agent Swarms:**
      * *Fundamental Analyst:* Deep value investing analysis.
      * *Technical Analyst:* Chart patterns and momentum indicators.
      * *SNC Analyst:* Regulatory credit risk grading (Shared National Credit).

-----

### 🛠️ Getting Started

#### Prerequisites

  * Python 3.10+
  * Node.js (Required for the Neural Dashboard UI)
  * API Keys: OpenAI (or compatible LLM provider), Neo4j (optional for full Knowledge Graph).

#### Quick Start Guide

1.  **Clone and Enter Repository**
    ```bash
    git clone https://github.com/adamvangrover/adam.git
    cd adam
    ```

2.  **Interactive Setup (Recommended)**
    Run the setup wizard to check dependencies, configure API keys (or Mock Mode), and launch the system.
    ```bash
    python3 scripts/setup_interactive.py
    ```

    *Alternatively, use the legacy launcher:*
    ```bash
    ./run_adam.sh
    ```

    *Or for manual setup:*
    ```bash
    pip install -e .           # Install as a package
    python core/main.py        # Run the engine
    ```

3.  **Configure API Keys (Optional but Recommended)**
    The launcher creates a `.env` file if missing. Edit it to add your keys:
    ```bash
    OPENAI_API_KEY=sk-...
    ```

4.  **Open Mission Control**
    *   **UI:** `http://localhost:80` (Docker) or `http://localhost:3000` (Local)
    *   **Neural Dashboard:** `showcase/index.html`

#### Developer Experience

We provide standard tooling for developers:

*   **Install Dependencies:** `make install`
*   **Run Tests:** `make test`
*   **Lint Code:** `make lint`
*   **CI/CD:** Automated testing via GitHub Actions is configured in `.github/workflows/ci.yml`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

-----

### 📂 Repository Structure

A high-level overview of the "Hive Mind" structure:

| Directory | Description |
| :--- | :--- |
| `core/engine/` | **The Brain.** Contains the cyclical graph, neuro-symbolic planner, and meta-orchestrator. |
| `core/agents/` | **The Workforce.** Specialized agents (Analyst, Risk, Legal, Industry Specialists). |
| `core/simulations/` | **The Simulator.** Modules for running stress tests (e.g., Fraud\_Detection, Stress\_Testing). |
| `core/data_processing/` | **The Stomach.** Universal Ingestor and data quality scrubbers. |
| `showcase/` | **The Face.** UI assets for the Neural Dashboard and demos. |
| `data/` | **The Memory.** Knowledge base (`knowledge_graph.json`), seeds, and artisanal training sets. |
| `config/` | **The DNA.** System configuration YAMLs and the Prompt Library. |
| `docs/` | **The Manual.** Comprehensive documentation and architecture visions. |

-----

### 📚 Resources & Documentation

  * **Architecture Vision:** [Adam v23.0 "Adaptive Hive" Vision](https://github.com/adamvangrover/adam/tree/main/docs/v23_architecture_vision.md)
  * **Pipeline Details:** [Gold Standard Data Pipeline]((https://github.com/adamvangrover/adam/tree/main/docs/GOLD_STANDARD_PIPELINE.md)
  * **API Reference:** [API Documentation](https://github.com/adamvangrover/adam/tree/main/docs/api.md)
  * **User Manual:** [Comprehensive User Guide](https://github.com/adamvangrover/adam/tree/main/docs/user_guide.md)
  * **Demo Guide:** [Showcase Walkthrough](https://github.com/adamvangrover/adam/tree/main/docs/SHOWCASE_GUIDE.md)
  * **Prompt Library:** [v23.5 Autonomous Analyst Prompt](https://github.com/adamvangrover/adam/tree/main/prompt_library/Adam_v23.5_System_Prompt.md)

-----

### 🤝 Contributing

We welcome contributions from the community\! Whether it's a new agent skill, a data connector, or a UI enhancement.

1.  Read our [Contribution Guidelines](https://github.com/adamvangrover/adam/tree/main).
2.  Fork the repo and create your branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes (`git commit -m 'Add some amazing feature'`).
4.  Push to the branch (`git push origin feature/amazing-feature`).
5.  Open a Pull Request.

-----

### 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

```
```

## ⚡ Modernization & Optimization (v23.5+)

As part of the 2025 Strategic Technical Modernization, the following high-performance components have been added:

*   **Modern Build System:** `pyproject.toml` and `uv` support for hermetic builds.
*   **Optimization Service:** A dedicated stateful microservice (`src/adam/api`) providing "Optimizer as a Service" via FastAPI and Redis.
*   **State-of-the-Art Optimizers:**
    *   **AdamW:** Decoupled Weight Decay.
    *   **Lion:** Evolved Sign Momentum (Google ADK).
    *   **Adam-mini:** Memory-efficient block-wise optimization (2025 Frontier).

### Quick Start (Modern Stack)

1.  **Build with Docker:**
    ```bash
    docker build -f Dockerfile.modern -t adam-optimizer .
    ```

2.  **Run the API:**
    ```bash
    docker run -p 8000:8000 -e REDIS_URL=redis://host.docker.internal:6379/0 adam-optimizer
    ```

For details, see [Modernization Report](docs/modernization_report.md).
