"""
Purpose: Host pure financial mathematical functions and deterministic logic algorithms.
Dependencies: typing
Outputs: calculate_leverage, check_covenant_compliance
"""

from typing import Dict, Any, List

def calculate_leverage(debt: float, ebitda: float) -> float:
    """
    Calculates Debt/EBITDA leverage ratio.
    """
    if ebitda == 0:
        return 999.9  # Avoid division by zero
    return debt / ebitda

def check_covenant_compliance(current_metric: float, covenant_limit: float, metric_type: str = "max") -> bool:
    """
    Checks if a metric complies with a covenant.
    metric_type: 'max' (e.g. Leverage < 5.0) or 'min' (e.g. Coverage > 1.2)
    """
    if metric_type == "max":
        return current_metric <= covenant_limit
    elif metric_type == "min":
        return current_metric >= covenant_limit
    return False
