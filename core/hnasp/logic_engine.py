from typing import Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

class LogicEngine:
    def __init__(self):
        pass

    def _get_var(self, data: Dict[str, Any], var_path: str):
        parts = var_path.split('.')
        curr = data
        for part in parts:
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                return None
        return curr

    def _evaluate(self, rule: Any, data: Dict[str, Any]) -> Any:
        if not isinstance(rule, dict):
            return rule

        # Keys are operators. JsonLogic usually has 1 key per object.
        # But if multiple keys, behavior is undefined/implementation dependent.
        # We assume 1 key.
        op = list(rule.keys())[0]
        values = rule[op]

        # Helper to ensure values is a list if it's not
        if not isinstance(values, list):
            values = [values]

        # Recursively evaluate arguments
        # Note: 'var' handles its own arg evaluation if needed, but usually var takes a string.
        # 'if' handles lazy evaluation.

        if op == "var":
            # var takes 1 argument: key, or 2: key, default
            # but arguments can be rules themselves? Usually var arg is a string.
            # But json logic spec allows {"var": ["a.b", "default"]}
            key = self._evaluate(values[0], data) if values else None
            default = self._evaluate(values[1], data) if len(values) > 1 else None

            val = self._get_var(data, str(key)) if key is not None else data
            return val if val is not None else default

        # Eager evaluation for other ops
        args = [self._evaluate(v, data) for v in values]

        if op == "==":
            return args[0] == args[1]
        elif op == "===":
            return args[0] == args[1]
        elif op == "!=":
            return args[0] != args[1]
        elif op == "!==":
            return args[0] != args[1]
        elif op == ">":
            return args[0] > args[1]
        elif op == ">=":
            return args[0] >= args[1]
        elif op == "<":
            return args[0] < args[1]
        elif op == "<=":
            return args[0] <= args[1]
        elif op == "and":
            return all(args)
        elif op == "or":
            return any(args)
        elif op == "!":
            return not args[0]
        elif op == "if":
            # Re-evaluate logic for 'if' to support lazy execution properly?
            # Actually, standard implementation eager evals the list first.
            # JsonLogic spec says 'if' is if/then/elseif/then/else.
            # {"if": [cond, true_val, false_val]}
            # My simple implementation above eager evaluated everything.
            # For correctness with side effects (none here) or performance, lazy is better.
            # But for this simple implementation, eager is fine.
            # The args are [cond, true, false].
            if args[0]:
                return args[1]
            return args[2] if len(args) > 2 else None
        else:
             # Unknown operator
             return None

    def validate_rule(self, rule: Dict[str, Any], data: Dict[str, Any]) -> Any:
        """
        Executes a JsonLogic rule against the provided data.
        """
        try:
            result = self._evaluate(rule, data)
            return result
        except Exception as e:
            logger.error(f"Logic execution failed: {e}")
            raise

    def batch_validate(self, rules: Dict[str, Dict[str, Any]], data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes multiple rules.
        """
        results = {}
        for rule_id, rule_def in rules.items():
            try:
                results[rule_id] = self.validate_rule(rule_def, data)
            except Exception as e:
                results[rule_id] = {"error": str(e)}
        return results

    def verify_trace(self, rule: Dict[str, Any], data: Dict[str, Any], claimed_result: Any) -> bool:
        """
        Verifies if the claimed result matches the actual execution.
        """
        actual_result = self.validate_rule(rule, data)
        return actual_result == claimed_result
