# How to Implement Capability Requests

This guide explains how to define and evaluate capabilities within the Adam OS vNext architecture using the `CapabilityEngine` and `PolicyEngine`.

## 1. Define the Capability
Capabilities are defined using the `CapabilityRequest` schema. This involves identifying the ID, the required risk class (e.g., `READ`, `FINANCIAL`, `ADMIN`), and linking it to a JSONLogic policy bundle.

```python
from src.pdil.authorization.capability_matrix import CapabilityRequest, RiskClass

req = CapabilityRequest(
    capability_id="portfolio.trade.execute",
    risk_class=RiskClass.FINANCIAL,
    policy_bundle="trade_execution_policy",
    approval=True,
    idempotency=True,
    provenance=True
)
```

## 2. Register with the Capability Engine
The `CapabilityEngine` enforces a strict deny-by-default posture. You must register your capability for it to be accessible.

```python
from src.pdil.authorization.capability_matrix import CapabilityEngine

engine = CapabilityEngine(default_deny=True)
engine.register_capability(req)
```

## 3. Evaluate Policy Constraints
Once a capability request passes the engine's baseline constraints (e.g., verifying that a FINANCIAL request has explicitly mapped approval), the specific JSONLogic policies are evaluated dynamically by the `PolicyEngine`.

```python
from src.pdil.authorization.policy_engine import PolicyEngine

policies = {
    "trade_execution_policy": {"==": [{"var": "context.user_role"}, "admin"]}
}
policy_engine = PolicyEngine(policies)

# Execution context during runtime
context = {"user_role": "admin"}
is_authorized = policy_engine.evaluate(req, context)
```

Following this guide ensures your agent integrations adhere to strict authorization boundaries, preventing unauthorized code execution and mitigating prompt injection risks.
