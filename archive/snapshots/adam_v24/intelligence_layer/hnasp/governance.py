from typing import Any, Dict
# Mocking json-logic-quibble if not available, or implementing basic logic
import json

class JsonLogicGovernance:
    """
    HNASP: Deterministic Governance via JsonLogic.
    """
    def __init__(self):
        self.rules: Dict[str, Dict] = {}
        self._load_constitution()

    def _load_constitution(self):
        """
        Loads the 'Constitution' - hard constraints.
        Appendix B: JsonLogic Rule for Credit Risk
        """
        self.rules["credit_approval"] = {
            "if": [
                { "and": [
                    { "<": [{ "var": "debt_to_income" }, 0.43] },
                    { ">": [{ "var": "credit_score" }, 680] },
                    { "!": { "var": "has_bankruptcy_history" } }
                ] },
                "APPROVE",
                { "if": [
                    { ">": [{ "var": "collateral_value" }, { "*": [{ "var": "loan_amount" }, 1.2] }] },
                    "APPROVE_WITH_COLLATERAL",
                    "REJECT"
                ] }
            ]
        }

    def evaluate(self, rule_name: str, data: Dict[str, Any]) -> Any:
        if rule_name not in self.rules:
            raise ValueError(f"Rule {rule_name} not found in Constitution.")

        rule = self.rules[rule_name]
        return self._apply_json_logic(rule, data)

    def _apply_json_logic(self, logic: dict, data: dict) -> Any:
        """
        Simple recursive implementation of JsonLogic for the demo.
        Supports: if, and, <, >, !, *, var
        """
        if isinstance(logic, list):
            return [self._apply_json_logic(l, data) for l in logic]

        if not isinstance(logic, dict):
            return logic # Primitive

        operator = list(logic.keys())[0]
        args = logic[operator]

        # Helper to simplify recursion
        def eval_arg(idx):
            if isinstance(args, list) and len(args) > idx:
                return self._apply_json_logic(args[idx], data)
            elif not isinstance(args, list) and idx == 0: # single arg
                return self._apply_json_logic(args, data)
            return None

        if operator == "var":
            var_name = args if not isinstance(args, list) else args[0]
            return data.get(var_name)

        if operator == "if":
            # if [condition, true_val, false_val] OR [cond, true, cond2, true2, default]
            condition = eval_arg(0)
            if condition:
                return eval_arg(1)
            elif len(args) > 2:
                return eval_arg(2)
            return None

        if operator == "and":
            return all(self._apply_json_logic(arg, data) for arg in args)

        if operator == "<":
            return eval_arg(0) < eval_arg(1)

        if operator == ">":
            return eval_arg(0) > eval_arg(1)

        if operator == "!":
            return not eval_arg(0)

        if operator == "*":
            return eval_arg(0) * eval_arg(1)

        raise NotImplementedError(f"Operator {operator} not implemented in mock engine.")
