# Generation Script Registry

The `scripts/` directory contains numerous standalone python scripts responsible for executing distinct workflows or generating static assets/dashboards. This registry serves as the master index explaining the generated HTML dashboards.

## Core Dashboards
- **`generate_daily_index.py`**: Parses daily HTML transmission files to extract metadata and JavaScript array structures (`const modules = [...]`). It outputs the interactive `showcase/adam_daily_hub.html` dashboard, which includes a dynamic masonry card grid and a 'gated insider access' UI mechanism via an authentication modal (`#authModal`).
- **`generate_deep_dive_reports.py`**: Generates quantitative deep dive reports for predefined tickers by fetching historical market data via `yfinance`. It outputs synthesized reports as a JSON object assigned to `window.DEEP_DIVE_DATA` in `showcase/js/deep_dive_data.js`, which is loaded by the predictive frontend dashboards.
- **`generate_comprehensive_index.py`**: Consolidates various data streams into a master index at `showcase/comprehensive_index.html`.
- **`generate_predictive_reports.py`**: Generates the 'Predictive Deep Dives' Cyberpunk/Bloomberg-styled HTML dashboard located at `showcase/predictive_deep_dives.html`, displaying distress reports, actionable ideas, and ML JSON blobs.

## System Maintenance
- **`daily_ritual.py`**: The automated recursive execution wrapper for Protocol ARCHITECT_INFINITE, driving the system's biological growth via an LLM.

*(Note: Always execute scripts from the repository root via `PYTHONPATH=. uv run python scripts/script_name.py`)*
