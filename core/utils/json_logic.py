from typing import Any, Dict, List, Union

def jsonLogic(tests: Any, data: Any = None) -> Any:
    """
    Safe implementation of JsonLogic compatible with Python 3.12.
    """
    # 1. Primitives
    if not isinstance(tests, dict):
        return tests

    # 2. Operator extraction
    # Handle single key dict
    if len(tests) != 1:
        # Not a rule, treat as literal dict?
        # JsonLogic rules are strictly {op: args}
        # But if recursion passed a dict that isn't a rule...
        # Standard says "If it's an object (dictionary) ... and has exactly one key..."
        return tests

    op = list(tests.keys())[0]
    values = tests[op]

    # Helper for recursion
    def evaluate(val):
        return jsonLogic(val, data)

    # 3. Operators
    if op == "var":
        # var logic
        key = values[0] if isinstance(values, list) and len(values) > 0 else (values if not isinstance(values, list) else None)
        default = values[1] if isinstance(values, list) and len(values) > 1 else None

        if key is None or key == "":
            return data

        # Dot notation support
        try:
            current = data
            for part in str(key).split('.'):
                if isinstance(current, dict):
                    if part in current:
                        current = current[part]
                    else:
                        return default
                elif isinstance(current, list) and part.isdigit():
                    idx = int(part)
                    if 0 <= idx < len(current):
                        current = current[idx]
                    else:
                        return default
                else:
                    return default
            return current
        except Exception:
            return default

    # Recursive evaluation of arguments
    # Some operators (like 'if') evaluate lazily, but standard JsonLogic executes strictly?
    # Actually 'if' is conditional.

    if op == "if":
        # values should be a list
        if not isinstance(values, list): values = [values]

        # [cond, true, false]
        # [cond, true, cond, true, ..., false]

        for i in range(0, len(values) - 1, 2):
            cond = evaluate(values[i])
            if cond:
                return evaluate(values[i+1])

        if len(values) % 2 == 1:
            return evaluate(values[-1])
        return None

    # Eager evaluation for others
    args = []
    if isinstance(values, list):
        args = [evaluate(v) for v in values]
    else:
        args = [evaluate(values)]

    # Comparison
    if op == "==": return args[0] == args[1]
    if op == "===": return args[0] == args[1]
    if op == "!=": return args[0] != args[1]
    if op == "!==": return args[0] != args[1]
    if op == ">": return args[0] > args[1]
    if op == ">=": return args[0] >= args[1]
    if op == "<": return args[0] < args[1]
    if op == "<=": return args[0] <= args[1]

    # Logic
    if op == "!": return not args[0]
    if op == "!!": return bool(args[0])
    if op == "or": return any(args)
    if op == "and": return all(args)

    # Arithmetic
    if op == "+": return sum(float(x or 0) for x in args)
    if op == "*":
        res = 1.0
        for x in args: res *= float(x or 0)
        return res
    if op == "-":
        if len(args) == 1: return -args[0]
        return args[0] - args[1]
    if op == "/": return args[0] / args[1]
    if op == "%": return args[0] % args[1]

    # Default fallback: return primitive
    return tests
