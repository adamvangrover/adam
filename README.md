# Adam v23.0: Your AI-Powered Partner

> **Note:** This document describes the current stable version of the Adam system (v21.0). For details on the next-generation architecture, please see the [Adam v23.0 "Adaptive Hive" Vision](./docs/v23_architecture_vision.md).

# Adam v23.0: The Adaptive Hive Mind
**System Status:** v23.0 (Active) | v21.0 (Stable)
**Mission:** Autonomous Financial Analysis & Adaptive Reasoning

Adam has evolved. v23.0 introduces the "Adaptive System" architecture, featuring:

*   **Cyclical Reasoning Graph:** A self-correcting neuro-symbolic engine.
*   **Neural Dashboard:** Real-time visualization of agent thought processes.
*   **Hybrid Architecture:** Combining v21's reliability with v22's speed and v23's intelligence.
*   **Gold Standard Data Pipeline:** A rigorous "universal ingestion" process that scrubs and certifies all system knowledge.

[**Launch Neural Dashboard**](./showcase/neural_dashboard.html)

> **Note:** For details on the original v21.0 architecture, please see the v21.0 Documentation.

(Welcome to Adam, the most advanced financial AI system yet! We've supercharged our capabilities with an expanded agent network, enhanced simulation workflows, and a more sophisticated knowledge base to deliver unparalleled financial analysis and investment insights.)

[**Explore the interactive demo here!**](./showcase/index.html)

## 🚀 Mission Control
[**Launch Neural Dashboard**](./showcase/neural_dashboard.html)
Monitor real-time agent reasoning, knowledge graph updates, and risk simulations.

## 🌟 Key Capabilities (v23.0)

### 1. Cyclical Reasoning Engine
Unlike traditional linear chains, Adam v23 uses a cyclical graph (built on LangGraph) to iterate on analysis.

*   **Draft -> Critique -> Refine:** The system critiques its own work and refines it until quality thresholds are met.
*   **Self-Correction:** Detects missing data or logical fallacies and automatically launches remedial tasks.

### 2. Neuro-Symbolic Planner
Combines the creativity of LLMs with the logical rigor of Knowledge Graphs.

*   **Path Discovery:** "Plan a path from Apple Inc. to Credit Risk considering Supply Chain Constraints."
*   **Traceability:** Every conclusion is back-linked to specific nodes in the PROV-O ontology.

### 3. Gold Standard Data Pipeline
A new "Universal Ingestor" ensures that every piece of data in the system is high-quality.

*   **Ingest & Scrub:** Recursively scans reports, prompts, code, and data.
*   **Conviction Scoring:** Automatically assesses the quality and "conviction" of data (0-100%).
*   **Unified Access:** All data is normalized into a standard JSONL format accessible by any agent.
*   [Read the Pipeline Documentation](./docs/GOLD_STANDARD_PIPELINE.md)

### 4. AI Partner v23.5 ("The Omniscient Analyst")
A full-spectrum autonomous analyst capable of deep credit analysis, valuation, and quantum risk modeling.

*   **Deep Dive Pipeline:** 5-phase execution from Entity Resolution to Strategic Synthesis.
*   **Portable Prompt:** `prompt_library/AOPL-v1.0/system_architecture/autonomous_financial_analyst_v23_5.md`
*   **Config:** `config/Adam_v23.5_Portable_Config.json`

## 🛠️ Getting Started

### Prerequisites
*   Python 3.10+
*   Node.js (for full UI dev)

### Quick Start

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/adamvangrover/adam.git
    cd adam
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Adam:**
    ```bash
    python scripts/run_adam.py
    ```

4.  **View the Showcase:**
    Open `showcase/index.html` in your browser.

## 📂 Repository Structure

*   `core/v23_graph_engine/`: The heart of the new system.
    *   `cyclical_reasoning_graph.py`: The self-correcting analysis loop.
    *   `neuro_symbolic_planner.py`: The logic for pathfinding in the KG.
    *   `meta_orchestrator.py`: The central brain routing tasks.
*   `core/data_processing/`: Data ingestion and standardisation.
    *   `universal_ingestor.py`: The Gold Standard Pipeline.
*   `showcase/`: The "Mission Control" UI assets.
*   `data/`: Knowledge base and artisanal training sets.
*   `docs/`: Comprehensive documentation.

## 📚 Documentation

*   [Adam v20.0 Implementation Plan](docs/v20.0)
*   [System Requirements](docs/REQUIREMENTS.md)
*   [User Guide](docs/user_guide.md)
*   [API Documentation](docs/api_docs.yaml)
*   [Contribution Guidelines](CONTRIBUTING.md)
*   [Showcase Guide](docs/SHOWCASE_GUIDE.md): Walkthrough of the demo.
*   [v23 Architecture Vision](docs/v23_architecture_vision.md): Deep dive into the "Adaptive Hive".
*   [Gold Standard Pipeline](docs/GOLD_STANDARD_PIPELINE.md): Data ingestion guide.

## 🤝 Contributing
Contributions are welcome! Please check [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License
MIT License. See LICENSE for details.
