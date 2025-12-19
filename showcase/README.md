# ADAM v23.5 Showcase (Static Archive)

## Overview
This directory contains the **Static Artifact Showcase** for the Adam Financial System. It is designed to be hosted directly on **GitHub Pages** (or any static web server) without requiring a Python backend.

### Key Features
*   **Zero-Dependency:** Runs entirely in the browser using HTML/CSS/JS.
*   **Simulation Engine:** `js/app.js` includes a simulation loop that generates fake market ticks, system vital signs, and agent logs to bring the UI to life even in "Offline Mode".
*   **Interactive Components:**
    *   **Trading Terminal:** Live order book and charting simulation.
    *   **Knowledge Graph:** Visualization of the 12,000+ node semantic network.
    *   **Deep Dive Analyst:** Interactive report viewer.
    *   **Deployment Console:** Simulated terminal environment.

## How to View
### Online (GitHub Pages)
1.  Navigate to `https://[your-username].github.io/[repo-name]/showcase/`.
2.  The system will automatically detect the environment and switch to **Archive Mode**.

### Local Development
1.  Open `showcase/index.html` in your browser.
2.  Or run a simple Python server:
    ```bash
    cd showcase
    python3 -m http.server 8000
    ```
    Then visit `http://localhost:8000`.

## Directory Structure
*   `index.html`: The main "Mission Control" dashboard.
*   `js/`: Core logic (`app.js`, `nav.js`) and static data (`mock_data.js`).
*   `css/`: Global styles (`style.css`).
*   `data/`: Raw JSON artifacts used by the data loader.

## "Apex Architect" Notes
> "The map is not the territory, but a broken map is a broken territory."

This static showcase serves as a **Holographic Projection** of the true Adam system. While the backend reasoning engines (Python/LangGraph) are dormant here, the frontend state management and visualization layers remain fully functional, demonstrating the system's capabilities to stakeholders without complex deployment requirements.
