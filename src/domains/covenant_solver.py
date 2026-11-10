"""System 2 Legal/Financial Solver (Catala + Z3 + OpenFisca pattern)."""
from __future__ import annotations

from src.domains.neuro_symbolic_ports import (
    CatalaCovenantRule,
    compute_vectorized_cads_waterfall,
    verify_covenant_and_lmt_safety,
)

try:
    import z3  # type: ignore[import-not-found]
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


def solve_smt_covenant_feasibility(
    base_ebitda: float,
    max_addback_pct: float,
    interest: float,
    amortization: float,
    min_dscr: float = 1.05,
) -> tuple[bool, float]:
    """
    Formulates DSCR covenant satisfaction as an SMT problem using Z3 when available,
    with an exact analytical fallback.
    Returns (is_satisfiable, max_achievable_dscr).
    """
    capped_addback = base_ebitda * min(0.20, max_addback_pct)
    max_ebitdar = base_ebitda + capped_addback
    total_debt_service = interest + amortization

    if total_debt_service <= 0:
        return True, 999.0

    if HAS_Z3:
        solver = z3.Solver()
        actual_dscr = z3.Real("actual_dscr")
        ebitdar_val = z3.RealVal(max_ebitdar)
        debt_service_val = z3.RealVal(total_debt_service)
        dscr_floor_val = z3.RealVal(min_dscr)

        # actual_dscr * debt_service_val <= ebitdar_val and actual_dscr >= dscr_floor_val
        solver.add(actual_dscr * debt_service_val <= ebitdar_val)
        solver.add(actual_dscr >= dscr_floor_val)

        is_sat = solver.check() == z3.sat
        achievable = max_ebitdar / total_debt_service
        return is_sat, achievable

    achievable = max_ebitdar / total_debt_service
    return (achievable >= min_dscr), achievable


__all__ = [
    "CatalaCovenantRule",
    "compute_vectorized_cads_waterfall",
    "verify_covenant_and_lmt_safety",
    "solve_smt_covenant_feasibility",
]
