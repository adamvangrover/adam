# Adam Sovereign Credit Bundle (v1.0.0)

This package abstracts the logic, schemas, and prompts of the "Adam Sovereign" system into a portable, modular format. It is designed for deployment in secure, air-gapped "Glass Box" environments (e.g., banking "Iron Bank" networks).

## Overview

The bundle contains the "Source Code" for the Adam Sovereign agents and governance rules, decoupled from the execution engine.

### Directory Structure

*   **`manifest.yaml`**: Configuration entry point defining security constraints and dependencies.
*   **`agents/`**: "Prompt-as-Code" definitions for specialized agents (Quant, Risk Officer, Archivist).
*   **`schemas/`**: JSON Schemas for data contracts (Sovereign Chunk, Audit Log).
*   **`governance/`**: Executable "Audit-as-Code" scripts (`adr_controls.py`) and golden datasets.
*   **`infra/`**: Terraform stubs for infrastructure isolation.

## Usage

### Prerequisites
*   Python 3.10+
*   `PyYAML`

### Running the Demo
A demonstration script is provided to verify the bundle's integrity and demonstrate the "Audit-as-Code" capabilities.

```bash
python scripts/demo_sovereign_bundle.py
```

This script will:
1.  Load the `manifest.yaml`.
2.  Instantiate the agents defined in `agents/`.
3.  Run the governance controls against the `golden_dataset.jsonl`.

### Running the Full Sovereign Pipeline
To simulate a full credit analysis pipeline using mock SEC EDGAR data for Mega Cap Tech (AAPL, MSFT, etc.):

1.  **Run the Pipeline:**
    ```bash
    python scripts/run_sovereign_pipeline.py
    ```
    This generates artifacts (Spreads, Memos, Audit Logs) in `showcase/data/sovereign_artifacts/`.

2.  **Launch the Dashboard:**
    Open `showcase/sovereign_dashboard.html` in a web browser (via a local server to avoid CORS issues).
    ```bash
    # From the repo root
    python3 -m http.server 8000
    # Navigate to http://localhost:8000/showcase/sovereign_dashboard.html
    ```

## Integration

To integrate this bundle into your orchestration framework (LangChain, AutoGen, etc.):

1.  **Load the Manifest:**
    ```python
    import yaml
    with open("enterprise_bundle/adam-sovereign-bundle/manifest.yaml") as f:
        config = yaml.safe_load(f)
    ```

2.  **Instantiate Agents:**
    Use a prompt loader to read the YAML files in `agents/` and inject the `system_prompt` into your LLM calls.

3.  **Run Governance Checks:**
    Import `governance/adr_controls.py` and run `audit_financial_math` or `audit_citation_density` on the agent outputs before returning them to the user.
