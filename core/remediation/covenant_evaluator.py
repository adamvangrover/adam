"""
Covenant Input Validation & Fail-Closed Covenant Evaluation Engine (F-02, F-07, Gate G0, Gate G2).

Enforces:
1. Strict missing-input rejection (never defaults missing secured debt or covenants to 0.0).
2. AST structure, operator arity, and numeric domain validation for jsonLogic expressions.
3. Fail-closed error handling for non-finite numbers (NaN, Inf) and domain violations (negative debt, zero EBITDA).
"""

import math


class CovenantValidationError(ValueError):
    """Raised when covenant inputs or rules fail schema, domain, or missingness checks."""
    pass


class JsonLogicASTError(ValueError):
    """Raised when jsonLogic expression AST is malformed, has invalid operator arity, or unsupported types."""
    pass


ALLOWED_JSONLOGIC_OPERATORS = {"<=", "<", ">=", ">", "==", "!=", "/", "*", "+", "-", "var"}


def validate_jsonlogic_ast(expr: object, valid_vars: set[str]) -> bool:  # noqa: C901
    """
    Validates a jsonLogic rule AST for allowed operators, correct arity, and variable names.
    """
    if isinstance(expr, (int, float)):
        if math.isnan(expr) or math.isinf(expr):
            raise JsonLogicASTError(f"Numeric literal '{expr}' is non-finite (NaN or Inf).")
        return True

    if isinstance(expr, (str, bool)):
        return True

    if isinstance(expr, dict):
        if len(expr) != 1:
            raise JsonLogicASTError("jsonLogic expression node must contain exactly one operator key.")

        op, args = next(iter(expr.items()))
        if op not in ALLOWED_JSONLOGIC_OPERATORS:
            raise JsonLogicASTError(f"Unsupported or dangerous jsonLogic operator: '{op}'")

        if op == "var":
            is_valid_var_arg = (
                isinstance(args, str)
                or (isinstance(args, list) and len(args) == 1 and isinstance(args[0], str))
            )
            if not is_valid_var_arg:
                raise JsonLogicASTError(f"Invalid var operator argument: {args}")
            var_name = args if isinstance(args, str) else args[0]
            if var_name not in valid_vars:
                raise JsonLogicASTError(f"Undeclared variable '{var_name}' in jsonLogic expression.")
            return True

        if not isinstance(args, list):
            raise JsonLogicASTError(f"Operator '{op}' expects a list of operands.")

        if op in {"<=", "<", ">=", ">", "==", "!=", "/", "*", "+", "-"}:
            if len(args) != 2:
                raise JsonLogicASTError(f"Operator '{op}' requires exactly 2 arguments, got {len(args)}.")

        for arg in args:
            validate_jsonlogic_ast(arg, valid_vars)

        return True

    raise JsonLogicASTError(f"Invalid jsonLogic AST node type: {type(expr)}")


def evaluate_covenant_leverage(inputs: dict[str, object]) -> dict[str, object]:  # noqa: C901
    """
    Evaluates total leverage and secured leverage covenants with strict fail-closed validation.

    Required fields in inputs:
    - total_debt (float >= 0)
    - senior_secured_debt (float >= 0)
    - ebitda (float > 0)
    - max_total_leverage_covenant (float > 0)
    - max_secured_leverage_covenant (float > 0)
    """
    required_fields = {
        "total_debt",
        "senior_secured_debt",
        "ebitda",
        "max_total_leverage_covenant",
        "max_secured_leverage_covenant",
    }

    missing_fields = required_fields - set(inputs.keys())
    if missing_fields:
        raise CovenantValidationError(
            f"Missing required covenant inputs: {sorted(list(missing_fields))}. "
            "Never substitute zero or default values for missing debt or covenant exposure."
        )

    for field in required_fields:
        val = inputs[field]
        if val is None:
            raise CovenantValidationError(f"Required input '{field}' is None. Explicit value required.")
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise CovenantValidationError(f"Field '{field}' must be a finite number, got {type(val)}.")
        if math.isnan(val) or math.isinf(val):
            raise CovenantValidationError(f"Field '{field}' contains non-finite value '{val}'.")

    total_debt = float(inputs["total_debt"])  # type: ignore[arg-type]
    senior_secured_debt = float(inputs["senior_secured_debt"])  # type: ignore[arg-type]
    ebitda = float(inputs["ebitda"])  # type: ignore[arg-type]
    max_total = float(inputs["max_total_leverage_covenant"])  # type: ignore[arg-type]
    max_secured = float(inputs["max_secured_leverage_covenant"])  # type: ignore[arg-type]

    if total_debt < 0:
        raise CovenantValidationError(f"Negative total_debt '{total_debt}' is invalid.")
    if senior_secured_debt < 0:
        raise CovenantValidationError(f"Negative senior_secured_debt '{senior_secured_debt}' is invalid.")
    if senior_secured_debt > total_debt:
        raise CovenantValidationError(
            f"Senior secured debt ({senior_secured_debt}) cannot exceed total debt ({total_debt})."
        )
    if ebitda <= 0:
        raise CovenantValidationError(f"EBITDA must be strictly positive for leverage calculation, got {ebitda}.")
    if max_total <= 0 or max_secured <= 0:
        raise CovenantValidationError("Covenant threshold leverage limits must be strictly positive.")

    # Calculations
    total_leverage = total_debt / ebitda
    secured_leverage = senior_secured_debt / ebitda

    total_leverage_headroom = max_total - total_leverage
    secured_leverage_headroom = max_secured - secured_leverage

    total_pass = total_leverage <= max_total
    secured_pass = secured_leverage <= max_secured
    compliant = total_pass and secured_pass

    return {
        "total_leverage": round(total_leverage, 4),
        "secured_leverage": round(secured_leverage, 4),
        "total_leverage_headroom": round(total_leverage_headroom, 4),
        "secured_leverage_headroom": round(secured_leverage_headroom, 4),
        "total_leverage_pass": total_pass,
        "secured_leverage_pass": secured_pass,
        "covenant_compliant": compliant,
        "status": "EVALUATED_FAIL_CLOSED",
    }
