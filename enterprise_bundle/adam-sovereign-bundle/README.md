# Adam Sovereign Credit Bundle (v1.0.0)

This bundle provides a comprehensive "Glass Box" architecture for automated credit memo generation, designed for high-compliance environments.

## Components

### Agents
- **Quant (`agents/quant.yaml`)**: Extracts financial data and validates accounting identities.
- **Risk Officer (`agents/risk_officer.yaml`)**: Checks for regulatory compliance and model risk.
- **Archivist (`agents/archivist.yaml`)**: Manages document retrieval and citation.
- **Writer (`agents/writer.yaml`)**: Synthesizes analysis into a professional credit memo.

### Governance
Located in `governance/adr_controls.py`.
- **Citation Density**: Enforces evidence linking (minimum 0.5 citations per sentence).
- **Financial Math**: Validates `Assets = Liabilities + Equity`.
- **Tone Check**: Detects unprofessional or emotive language.
- **Absolute Statements**: Flags risky absolute claims (e.g., "guaranteed").

### Schemas
Located in `schemas/`.
- **Sovereign Chunk**: Data contract for vector storage.
- **Audit Log**: Structure for compliance logging.
- **Credit Memo**: Output schema for the final report.

### Infrastructure
Located in `infra/`.
- **VPC Config**: Defines the "Iron Bank" isolation layer.
- **Outputs**: Exposes VPC and Subnet IDs.

## Usage

1. **Deploy Infrastructure**: Use Terraform to apply the configuration in `infra/`.
2. **Run Governance Tests**: Execute `python3 tests/test_governance.py` to verify control logic.
3. **Instantiate Agents**: Load agent definitions from `agents/` into your orchestration engine.

## Security

This bundle is designed for "Air-Gapped" or "Private Cloud" deployment.
- No public internet access for vector databases.
- Strict PII redaction requirements.
- synchronous audit logging.
