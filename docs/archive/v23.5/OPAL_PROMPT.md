# Adam v23.5 Development Prompt

**Project Title:** Adam v23.5 - Autonomous Financial Analysis & Adaptive Reasoning Platform

**Objective:**
Build a full-stack, agentic AI platform designed for autonomous financial research, risk assessment, and market simulation. The system, known as "Adam," must utilize a "Cyclical Reasoning Graph" architecture that allows it to draft, critique, and self-correct its own analysis before presenting results to the user.

**1. Core Philosophy & Architecture:**
* **System Type:** Adaptive Hive Mind / Neuro-Symbolic Engine.
* **Logic Model:** Move beyond linear chains to a graph-based execution model. Implement a **Cyclical Reasoning Engine** (`core/engine/cyclical_reasoning_graph.py`) that follows a `Draft -> Critique -> Refine` loop to ensure high conviction in generated insights.
* **Orchestration:**
    * **Meta Orchestrator:** Acts as the central cortex (`core/engine/meta_orchestrator.py`) to route tasks and manage state.
    * **Neuro-Symbolic Planner:** Combines LLM creativity with Knowledge Graph logic (`core/engine/neuro_symbolic_planner.py`) to plan execution paths.
* **Data Integrity:** Implement a **Gold Standard Data Pipeline** ("Universal Ingestor") that ingests, scrubs, and scores data for "conviction" (0-100%) from sources like SEC filings (XBRL), news APIs, and market feeds.

**2. Tech Stack Specification:**

* **Frontend (Mission Control Dashboard):**
    * **Framework:** React 19 (`react`, `react-dom`) with `react-scripts`.
    * **Styling:** Tailwind CSS (`tailwindcss`, `tailwind-merge`, `clsx`) for a modern, glass-morphism aesthetic.
    * **Visualization:**
        * `chart.js` and `react-chartjs-2` for financial data plotting.
        * `react-force-graph-2d` for real-time visualization of the Knowledge Graph and agent thought processes.
    * **State/Network:** `axios` for API calls, `socket.io-client` for real-time agent updates.
    * **Icons:** `lucide-react`.
    * **Routing:** `react-router-dom`.

* **Backend (The Brain):**
    * **Language:** Python 3.10+.
    * **Core Libraries:** Semantic Kernel (for agent skills), Celery (for async task queueing), RabbitMQ (message broker).
    * **Database:** Neo4j (Unified Knowledge Graph) for storing relationships between financial entities, and a Vector Store for semantic retrieval.
    * **AI Models:** Integration with OpenAI API (or compatible LLM providers).

**3. Key Features to Implement:**
* **Neural Dashboard:** A real-time UI that visualizes the "Graph" of agent reasoning. Users should see nodes lighting up as agents (e.g., Fundamental Analyst, Risk Agent, Legal Agent) execute tasks.
* **Multi-Agent Swarms:**
    * **Fundamental Analyst:** Performs deep value investing analysis (DCF models, peer comparison).
    * **Technical Analyst:** Analyzes chart patterns and momentum.
    * **SNC Analyst:** Conducts regulatory credit risk grading.
    * **Risk Architect:** Runs Monte Carlo simulations and stress tests.
* **Traceability:** All generated insights must be back-linked to source documents (PROV-O compliant) in the Knowledge Graph.
* **Reporting:** Output structured artifacts including JSON deep dives, PDF reports, and interactive HTML dashboards.

**4. Directory Structure Blueprint:**
* `core/engine/`: Logic for the planner, graph engine, and orchestrator.
* `core/agents/`: Definitions for specialized agents (Analyst, Risk, Legal).
* `core/simulations/`: Modules for stress testing and black swan scenarios.
* `core/data_processing/`: The Universal Ingestor pipeline.
* `services/webapp/`: The React frontend and API backend.
* `data/`: Storage for the Knowledge Base and artisanal training sets.
