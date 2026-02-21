# 🎓 Office Nexus Tutorial

Welcome to the **Office Nexus**, the Adam v26.0 Desktop Environment. This interface provides a simulated "Operating System" for interacting with financial data, reports, and system diagnostics.

## 🚀 Getting Started

To launch Office Nexus, open `showcase/index.html` (which redirects to `office_nexus.html`) in your browser.

When the system boots, you will see a desktop environment with icons, a taskbar, and a start menu.

## 🖥️ The Desktop

The desktop contains shortcuts to frequently used applications:

*   **My Computer**: Browse the repository file system.
*   **Market Monitor**: Real-time view of S&P 500 data (Prices, P/E, Ratings).
*   **Credit Sentinel**: Monitor credit risk scores and PD/LGD metrics.
*   **Report Generator**: Generate new Equity Reports or Credit Memos.
*   **System Health**: Monitor backend system status (CPU, RAM, Agents).
*   **Showcase**: Browse the generated HTML artifacts.

## 📱 Using Apps

### Market Monitor
Click the **Market Monitor** icon to see a live table of market data.
*   **Sort**: The data is sorted by Ticker by default.
*   **Drill-down**: Click on any row to open the detailed Equity Report for that company.

### Credit Sentinel
This app visualizes risk.
*   **Green**: Low Risk (Investment Grade)
*   **Orange**: Medium Risk (High Yield / Crossover)
*   **Red**: High Risk (Distressed)
*   Click on a card to view the detailed Credit Memo.

### System Health (New!)
Check the pulse of the Adam backend.
*   **CPU/Memory**: Live gauges of system resource usage.
*   **Active Agents**: Status of the backend agents (e.g., FinancialAgent, NewsAgent).
*   *Note: If running in static mode, these metrics are simulated.*

### Documentation Viewer
You are reading this file in the Documentation Viewer!
*   Use the **Reload** button to refresh the content if you've made changes to the Markdown file.

### Nexus Terminal (New!)
A fully interactive command-line interface.
*   Launch it from the Start Menu or Desktop.
*   **Commands**:
    *   `help`: List available commands.
    *   `ls`: List files in the current directory.
    *   `cat [filename]`: View file contents.
    *   `price [ticker]`: Check live market price (e.g., `price AAPL`).
    *   `peers [ticker]`: Identify competitor peer set (e.g., `peers NVDA`).
    *   `analyze [ticker]`: Trigger an agent analysis workflow.
    *   `clear`: Clear the screen.

## 📂 File Explorer

The **Explorer** app mimics a standard file manager.
*   Navigate through folders by double-clicking.
*   Open files (HTML, JSON, TXT) by double-clicking.
*   HTML files open in the **Browser**.
*   JSON/CSV files open in the **Spreadsheet**.
*   Text files open in **Notepad**.

## 🛠️ Troubleshooting

If apps fail to load data:
1.  Ensure you have run the data generation scripts:
    ```bash
    python scripts/generate_sp500_micro_build.py
    python scripts/generate_filesystem_manifest.py
    ```
2.  Check the browser console (F12) for errors.

---
*Adam v26.0 - The Neuro-Symbolic Financial Sovereign*
