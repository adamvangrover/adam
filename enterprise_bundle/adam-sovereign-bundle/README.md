Here is the merged documentation for the **Adam Sovereign Credit Bundle (v1.0.0)**. It combines the architectural details and specific component breakdowns from the `main` branch with the practical usage, demo, and integration steps from the feature branch.

---

# Adam Sovereign Credit Bundle (v1.0.0)

This package abstracts the logic, schemas, and prompts of the "Adam Sovereign" system into a portable, modular format. It provides a comprehensive **"Glass Box"** architecture for automated credit memo generation, designed for high-compliance environments and deployment in secure, air-gapped "Iron Bank" networks.

## Overview

The bundle contains the "Source Code" for the Adam Sovereign agents and governance rules, decoupled from the execution engine. It enforces "Prompt-as-Code" and "Audit-as-Code" paradigms.

### Architecture & Components

* **`manifest.yaml`**: Configuration entry point defining security constraints, dependencies, and bundle metadata.
* **`agents/`**: "Prompt-as-Code" definitions for specialized agents.
* **Quant (`agents/quant.yaml`)**: Extracts financial data and validates accounting identities.
* **Risk Officer (`agents/risk_officer.yaml`)**: Checks for regulatory compliance and model risk.
* **Archivist (`agents/archivist.yaml`)**: Manages document retrieval and citation.
* **Writer (`agents/writer.yaml`)**: Synthesizes analysis into a professional credit memo.


* **`governance/`**: Executable "Audit-as-Code" scripts (`adr_controls.py`) and golden datasets. Key controls include:
* **Citation Density**: Enforces evidence linking (minimum 0.5 citations per sentence).
* **Financial Math**: Validates `Assets = Liabilities + Equity`.
* **Tone Check**: Detects unprofessional or emotive language.
* **Absolute Statements**: Flags risky absolute claims (e.g., "guaranteed").


* **`schemas/`**: JSON Schemas for data contracts.
* **Sovereign Chunk**: Data contract for vector storage.
* **Audit Log**: Structure for compliance logging.
* **Credit Memo**: Output schema for the final report.


* **`infra/`**: Terraform stubs for infrastructure isolation.
* **VPC Config**: Defines the "Iron Bank" isolation layer.
* **Outputs**: Exposes VPC and Subnet IDs.



## Security

This bundle is designed for **"Air-Gapped"** or **"Private Cloud"** deployment to ensure data sovereignty.

* **Network Isolation**: No public internet access for vector databases; relies on local or private endpoints.
* **PII Handling**: Strict PII redaction requirements baked into agent prompts.
* **Auditability**: Synchronous audit logging for all agent actions.

## Usage

### Prerequisites

* Python 3.10+
* `PyYAML`
* Terraform (if deploying infrastructure)

### 1. Infrastructure Deployment

Use Terraform to apply the configuration in `infra/` to set up the isolated "Iron Bank" environment.

```bash
terraform apply -chdir=infra/

```

### 2. Verification

Before running the full pipeline, verify the bundle's integrity and governance logic.

**Run Governance Unit Tests:**
Execute the specific governance logic tests to ensure controls (Math, Tone, etc.) are functioning.

```bash
python3 tests/test_governance.py

```

**Run the Sovereign Demo:**
This script loads the manifest, instantiates agents, and runs controls against the `golden_dataset.jsonl`.

```bash
python scripts/demo_sovereign_bundle.py

```

### 3. Running the Full Sovereign Pipeline

To simulate a complete credit analysis pipeline using mock SEC EDGAR data for Mega Cap Tech (AAPL, MSFT, etc.):

1. **Run the Pipeline:**
This generates artifacts (Spreads, Memos, Audit Logs) in `showcase/data/sovereign_artifacts/`.
```bash
python scripts/run_sovereign_pipeline.py

```


2. **Launch the Dashboard:**
Open `showcase/sovereign_dashboard.html` in a web browser. Use a local server to avoid CORS issues.
```bash
# From the repo root
python3 -m http.server 8000
# Navigate to http://localhost:8000/showcase/sovereign_dashboard.html

```



## Integration

To integrate this bundle into your orchestration framework (LangChain, AutoGen, etc.):

1. **Load the Manifest:**
```python
import yaml
with open("enterprise_bundle/adam-sovereign-bundle/manifest.yaml") as f:
    config = yaml.safe_load(f)

```


2. **Instantiate Agents:**
Use a prompt loader to read the YAML files in `agents/` and inject the `system_prompt` into your LLM calls.
3. **Run Governance Checks:**
Import `governance/adr_controls.py` and run specific audits on agent outputs before returning them to the user.
```python
from governance.adr_controls import audit_financial_math, audit_citation_density

# Example check
audit_result = audit_financial_math(agent_output)

```