# Portable Content Index

This index lists all "Portable Assets" within the Adam v23.5 repository. These files are designed to be self-contained, allowing for instant deployment of specific personas ("Brains") and visualization interfaces ("Mission Control") without complex backend dependencies.

## Portable Configurations ("Brains")

The "Brain" of the agent—its role, directive, execution protocol, and specialized knowledge—is encapsulated in a single JSON file. This allows for instant "Hot-Swapping" of capabilities.

| File Path | Description | Key Capabilities |
| :--- | :--- | :--- |
| `config/Adam_v23.5_Portable_Config.json` | **The AI Partner Upgrade** | **Deep Credit & Valuation**: Full-spectrum analyst capable of DCF, SNC Ratings, and Quantum Risk Modeling. |
| `config/Adam_v22.0_Portable_Config.json` | The v22.0 Enterprise Base | **Auditability & Compliance**: Focused on PROV-O data provenance and regulatory reporting. |

## Portable Prompts ("Personas")

The raw instructional material that drives the agent's reasoning. These markdown files can be fed directly into any LLM context window.

| File Path | Description |
| :--- | :--- |
| `prompt_library/Adam_v23.5_System_Prompt.md` | **The Master Prompt**: The complete instructional set for the v23.5 "Omniscient Analyst", including the 5-Phase Deep Dive Protocol. |
| `prompt_library/AOPL-v1.0/` | **AOPL Library**: The "Agent Oriented Programming Language" modules, breaking down specific skills (e.g., `credit_analysis.md`, `market_sentiment.md`). |

## Static Dashboards ("Mission Control")

These HTML interfaces are **Client-Side Portable**. They are located in the `showcase/` directory and can be opened directly in any modern web browser to visualize the system's output. They do not strictly require a backend server for demonstration, as they can load mock data (`showcase/js/mock_data.js`).

### Dashboard Gallery

#### 1. Neural Dashboard (`showcase/neural_dashboard.html`)
*   **Purpose:** Real-time visualization of the "Hive Mind" and agent reasoning processes.
*   **Key Features:**
    *   **Live Thought Trace:** Watch the Neuro-Symbolic Planner decompose queries.
    *   **System Health:** Monitor active agents and resource usage.
    *   **Cyber-Minimalist UI:** Glassmorphism and data-dense layouts.

#### 2. Deep Dive Explorer (`showcase/deep_dive.html`)
*   **Purpose:** Interactive report viewer for the "Deep Dive" protocol (v23.5).
*   **Key Features:**
    *   **Market Radar:** Sector-wide scanning and heatmaps.
    *   **Report Library:** Searchable index of generated analysis artifacts.
    *   **Multi-Tab Analysis:** Switch between Entity Ecosystem, Valuation, and Risk views.

#### 3. Financial Digital Twin (`showcase/financial_twin.html`)
*   **Purpose:** Entity-specific visualization.
*   **Key Features:**
    *   **Knowledge Graph Visualization:** Interactive node-link diagrams of supply chains and ownership.
    *   **Financial Metrics:** Dynamic charts of revenue, EBITDA, and debt.

---

**Key Insight:** These static dashboards demonstrate the "Decoupled UI" architecture. The backend (Adam) generates standardized JSON artifacts (HDKG), and the frontend (Showcase) simply renders them. This separation ensures that the analysis engine can run anywhere (serverless, on-prem container), while the results remain universally accessible.
