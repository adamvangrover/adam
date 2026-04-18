# Generation Script Registry

The `scripts/` directory contains numerous standalone python scripts responsible for executing distinct workflows or generating static assets/dashboards.

## Core Dashboards
- **`generate_daily_index.py`**: Parses daily brief HTML files to extract metadata and outputs the interactive `showcase/adam_daily_hub.html`. This dashboard includes a masonry card grid and a 'gated insider access' UI mechanism with an authentication modal (`#authModal`).
- **`generate_comprehensive_index.py`**: Consolidates various data streams into a master index at `showcase/comprehensive_index.html`.
- **`generate_predictive_reports.py`**: Generates the 'Predictive Deep Dives' Cyberpunk/Bloomberg-styled HTML dashboard located at `showcase/predictive_deep_dives.html`, displaying distress reports, actionable ideas, and ML JSON blobs.

## System Maintenance
- **`daily_ritual.py`**: The automated recursive execution wrapper for Protocol ARCHITECT_INFINITE.

*(Note: Always execute scripts from the repository root via `PYTHONPATH=. uv run python scripts/script_name.py`)*
