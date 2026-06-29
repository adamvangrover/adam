# How-To: Configure JSONLogic Covenants

ADAM decouples business logic from stochastic AI execution using JSONLogic.

## 1. Define the Rule
Create a rule in `config/covenants/` to define a threshold.

```json
{
  "covenant_name": "Max Leverage Ratio",
  "rule": {
    "<=": [ { "var": "calculated_leverage" }, 4.5 ]
  }
}
```

## 2. Triggering the Rule
When the `QuantAgent` outputs a ratio, it is passed through the `JsonLogicGovernanceGatekeeper` which runs the rule deterministically against the extracted variables.
