import logging
from typing import Any, Dict

from core.engine.state_machine import FinancialState

logger = logging.getLogger(__name__)

def financial_validation_reflector(state: FinancialState) -> Dict[str, Any]:
    """
    LangGraph Reflector Node: Validates DCF and simulation outputs to ensure
    they adhere to strict financial constraints.
    """
    logger.info("Executing Financial Validation Reflector...")

    simulation_results = state.get("simulation_results", {})
    financial_context = state.get("financial_context", {})
    iteration_count = state.get("iteration_count", 0)

    critique_notes = []
    is_valid = True

    # 1. Validation: Ensure we actually have simulation paths
    if "stress_test" not in simulation_results:
        critique_notes.append("Missing stress test results in simulation output.")
        is_valid = False

    # 2. Validation: Check for invalid VaR (Value at Risk cannot be positive)
    stress_test = simulation_results.get("stress_test", {})
    var_95 = stress_test.get("var_95")
    if var_95 is not None and var_95 > 0:
        critique_notes.append(f"Value at Risk (95%) is positive ({var_95}), which violates constraint (should be <= 0).")
        is_valid = False

    # 3. Validation: DCF Context constraints
    # E.g. Check if the terminal growth rate exceeds the discount rate
    if financial_context:
        wacc = financial_context.get("wacc", 0.08)
        tgr = financial_context.get("terminal_growth_rate", 0.02)
        if tgr >= wacc:
            critique_notes.append(f"Terminal growth rate ({tgr}) must be strictly less than WACC ({wacc}).")
            is_valid = False

    # Check max iterations to force exit and avoid infinite loops
    if not is_valid and iteration_count >= 3:
        logger.warning("Max iteration count reached. Forcing valid state despite violations.")
        critique_notes.append("[WARNING] Max iterations reached. Proceeding with known constraints violations.")
        is_valid = True

    # Return updates to the state
    return {
        "is_valid": is_valid,
        "critique_notes": critique_notes,
        "iteration_count": iteration_count + 1
    }

def should_recalculate(state: FinancialState) -> str:
    """
    Conditional edge function used by LangGraph to determine the next step.
    """
    if state.get("is_valid", False):
        return "finalize"
    return "recalculate"
