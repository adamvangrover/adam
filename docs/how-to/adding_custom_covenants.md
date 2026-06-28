# How-To: Adding Custom Covenants

This guide explains how to translate a new credit agreement clause into a JSONLogic rule for the `adam-credit` module.

## Understanding Logic as Data
In ADAM, underwriting policies and covenants are decoupled from code and defined as "Logic as Data" using `jsonLogic`. This ensures deterministic risk evaluation.

## Step 1: Define the Clause
Suppose you have a covenant: "Total Leverage Ratio must not exceed 4.0x."

## Step 2: Create the JSONLogic Rule
Create a JSON rule that represents this constraint.

```json
{
  "covenant_name": "Maximum Total Leverage",
  "rule": {
    "<=": [
      {"var": "total_leverage_ratio"},
      4.0
    ]
  }
}
```

## Step 3: Integrate with adam-credit
Save this rule in the repository's rules configuration directory, typically `config/covenants/max_leverage.json`.

During DAG execution, the Neuro-Symbolic Planner will load this JSONLogic to evaluate the extracted parameters from the 10-K against this deterministic threshold.

Always remember: When evaluating `jsonLogic` in Python, use the `json-logic-qubit` library (`from json_logic import jsonLogic`). Do not hallucinate internal kernel paths.
