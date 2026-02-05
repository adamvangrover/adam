# Configuration Guide

Adam v26.0 uses a tiered configuration system to ensure security and flexibility.

## 1. Hierarchy

1.  **Environment Variables (`.env`):** Highest priority. Used for **Secrets** (API Keys, Passwords).
2.  **Configuration Files (`config/*.yaml`):** Static configuration. Used for **System Behavior** (Agent roles, Logging levels).
3.  **Defaults (Code):** Fallback values hardcoded in Python.

## 2. Key Files

### `config.yaml`
The master configuration file.
*   **`active_agents`**: List of agents enabled at startup.
*   **`system_mode`**: `DEV`, `TEST`, or `PROD`.

### `agents.yaml`
Defines the specific parameters for each agent.
*   **`model`**: Which LLM to use (e.g., `gpt-4-turbo`, `claude-3-opus`).
*   **`temperature`**: Creativity setting (0.0 for Risk, 0.7 for Narrative).

### `governance_policy.yaml`
**Critical Security File.**
Defines the "Rules of Engagement" for the Agentic Oversight Framework (AOF).
*   **`max_trade_size`**: Limits the dollar amount an agent can propose.
*   **`prohibited_sectors`**: ESG exclusions (e.g., "Tobacco", "Weapons").

### `logging.yaml`
Controls the verbosity and format of system logs (Console vs File vs JSON).

## 3. Secrets Management
**NEVER commit secrets to `config/`.**
Always use the `.env` file. See `.env.example` for the required keys.
