# Zero to Hero: Setup Guide

Welcome to the **Adam v23.5** setup guide. This document will take you from a fresh clone to a running "Autonomous Financial Analyst" in minutes.

---

## 1. Prerequisites

Before starting, ensure your environment meets the following requirements:

*   **Operating System:** Linux, macOS, or Windows (WSL2 recommended).
*   **Python:** Version **3.10+** (Required for modern type hinting and async features).
*   **Node.js:** (Optional) Required only if you plan to rebuild the React frontend source code. The static `showcase/` dashboards work out-of-the-box.
*   **API Keys:** You will need keys for:
    *   **OpenAI:** (Core LLM reasoning)
    *   **Neo4j:** (Knowledge Graph storage - Optional for local mock mode)

---

## 2. Installation & Configuration

### Option A: Local Python Install (Recommended for Development)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-repo/adam.git
    cd adam
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    Set up your API keys. You can do this by creating a `.env` file in the root directory.

    **Example `.env`:**
    ```ini
    OPENAI_API_KEY=sk-your-key-here
    NEO4J_URI=bolt://localhost:7687
    NEO4J_PASSWORD=your-password
    # Optional: Enable specific graph features
    USE_V23_GRAPH=true
    ```

### Option B: Docker Container (Recommended for Deployment)

1.  **Build and Run:**
    ```bash
    docker-compose up --build
    ```
    This will start the Core Agent, Redis, and Neo4j (if configured) in isolated containers.

---

## 3. Launching the System

### Backend (The Brain)

To start the main execution loop, run the `run_adam.py` script. This initializes the Meta Orchestrator and waits for user input.

```bash
python scripts/run_adam.py
```

*   **Interactive Mode:** You can type queries directly (e.g., *"Analyze Apple Inc."*).
*   **Single Shot:** You can pass a query as an argument:
    ```bash
    python scripts/run_adam.py --query "Assess the credit risk of Tesla"
    ```

### Frontend (Mission Control)

To visualize the system's output, simply open the **Static Mission Control** dashboard in your browser. No backend server is strictly required for the UI to render the pre-generated showcase data.

*   **Open:** `showcase/index.html`
*   **Navigate:** Use the sidebar to explore the **Neural Dashboard**, **Deep Dive Explorer**, and **Financial Twin**.

---

## 4. Verification

To ensure that the "Workforce" of agents is correctly instantiated and communicating, run the verification script.

```bash
python scripts/test_new_agents.py
```

**What to look for:**
*   **"Orchestrator instantiated successfully"**: Confirms the configuration was loaded.
*   **Agent Output**: You should see JSON output from the `BehavioralEconomicsAgent` and `MetaCognitiveAgent`, confirming that the specific sub-agents are active and processing data.

---

## 5. Troubleshooting & FAQs

### Common Issues

**Q: I see a `Knowledge base file not found` error.**
**A:** Ensure that `data/risk_rating_mapping.json` exists. This file is critical for the SNC Rating Agent. A default one is usually created during the build process, but you may need to check the `data/` directory.

**Q: `facebook-scraper` module is missing.**
**A:** This dependency was removed in v23.0 due to conflicts with `semantic-kernel`. Social media ingestion will gracefully degrade (skip those steps) if the package is missing. This is expected behavior.

**Q: The UI isn't loading data.**
**A:** If you are running `showcase/*.html` directly from the file system, modern browsers might block CORS requests. Try running a simple local server:
```bash
python -m http.server
```
Then navigate to `http://localhost:8000/showcase/index.html`.

**Q: How do I run the "Deep Dive" specifically?**
**A:** In the interactive prompt, ensure your query is complex enough to trigger the routing logic (e.g., *"Perform a deep dive analysis on Microsoft"*). Simple queries like *"What is the stock price?"* will be routed to the faster, legacy v21 engine.
