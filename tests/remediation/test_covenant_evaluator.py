import pytest
import math
from core.remediation.covenant_evaluator import (
    evaluate_covenant_leverage,
    validate_jsonlogic_ast,
    CovenantValidationError,
    JsonLogicASTError,
)

def test_valid_covenant_evaluation():
    inputs = {
        "total_debt": 1800.0,
        "senior_secured_debt": 1125.0,
        "ebitda": 450.0,
        "max_total_leverage_covenant": 4.50,
        "max_secured_leverage_covenant": 3.00,
    }
    res = evaluate_covenant_leverage(inputs)
    assert res["total_leverage"] == 4.00
    assert res["secured_leverage"] == 2.50
    assert res["total_leverage_headroom"] == 0.50
    assert res["secured_leverage_headroom"] == 0.50
    assert res["covenant_compliant"] is True

def test_missing_secured_debt_raises_error():
    inputs = {
        "total_debt": 1800.0,
        # "senior_secured_debt" is missing!
        "ebitda": 450.0,
        "max_total_leverage_covenant": 4.50,
        "max_secured_leverage_covenant": 3.00,
    }
    with pytest.raises(CovenantValidationError, match="Missing required covenant inputs.*senior_secured_debt"):
        evaluate_covenant_leverage(inputs)

def test_missing_covenant_threshold_raises_error():
    inputs = {
        "total_debt": 1800.0,
        "senior_secured_debt": 1125.0,
        "ebitda": 450.0,
        "max_total_leverage_covenant": 4.50,
        # max_secured_leverage_covenant missing
    }
    with pytest.raises(CovenantValidationError):
        evaluate_covenant_leverage(inputs)

def test_zero_or_negative_ebitda_raises_error():
    inputs = {
        "total_debt": 1800.0,
        "senior_secured_debt": 1125.0,
        "ebitda": 0.0,
        "max_total_leverage_covenant": 4.50,
        "max_secured_leverage_covenant": 3.00,
    }
    with pytest.raises(CovenantValidationError, match="EBITDA must be strictly positive"):
        evaluate_covenant_leverage(inputs)

def test_nan_or_inf_raises_error():
    inputs = {
        "total_debt": math.nan,
        "senior_secured_debt": 1125.0,
        "ebitda": 450.0,
        "max_total_leverage_covenant": 4.50,
        "max_secured_leverage_covenant": 3.00,
    }
    with pytest.raises(CovenantValidationError, match="non-finite value"):
        evaluate_covenant_leverage(inputs)

def test_jsonlogic_ast_validation():
    valid_vars = {"total_leverage", "max_total_leverage"}
    rule = {"<=": [{"var": "total_leverage"}, {"var": "max_total_leverage"}]}
    assert validate_jsonlogic_ast(rule, valid_vars) is True

    # Malformed rule: invalid operator
    rule_invalid_op = {"$eval": [1, 2]}
    with pytest.raises(JsonLogicASTError, match="Unsupported or dangerous jsonLogic operator"):
        validate_jsonlogic_ast(rule_invalid_op, valid_vars)

    # Arity error
    rule_arity = {"<=": [1]}
    with pytest.raises(JsonLogicASTError, match="requires exactly 2 arguments"):
        validate_jsonlogic_ast(rule_arity, valid_vars)
