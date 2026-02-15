# Governance & Compliance

The Adam Financial Intelligence System includes a robust governance layer to ensure agent actions are ethical, compliant, and risk-managed.

## Overview

The governance model consists of:
1.  **Environment Control**: Managing deployments across environments (Dev, QA, Staging, Prod).
2.  **Constitution**: Defining core principles and rules for agent behavior.
3.  **MCP Integration**: Exposing governance checks via the MCP server.

## Constitution

The `Constitution` class (in `core/governance/constitution.py`) enforces principles like:
- **DO_NO_HARM**: Avoid actions causing financial ruin.
- **TRANSPARENCY**: Log all actions.
- **COMPLIANCE**: Adhere to regulations.
- **RISK_MANAGEMENT**: Limit risk exposure.

### Usage
The `check_governance_compliance` MCP tool allows external systems to verify actions against the constitution.

```python
import json
from server.server import check_governance_compliance

result = check_governance_compliance("EXECUTE_TRADE", json.dumps({"risk_score": 0.9}))
print(result)
```

## Logging & Auditing

All governance checks are logged using `core/utils/logger.py`. Audit logs can be retrieved via the `audit_log` attribute of the `Constitution` instance.

## Best Practices

- Always check governance before executing critical actions.
- Use `EnvironmentGate` for deploying new models or configurations.
- Regularly review audit logs for compliance violations.
