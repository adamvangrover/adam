"""System 2 Legal/Financial Solver (catala + z3 + openfisca + finos/cdm bridge)."""
from __future__ import annotations

from src.domains.neuro_symbolic_ports import (
    CatalaCovenantRule,
    compute_vectorized_cads_waterfall,
    verify_covenant_and_lmt_safety,
)

__all__ = [
    "CatalaCovenantRule",
    "compute_vectorized_cads_waterfall",
    "verify_covenant_and_lmt_safety",
]
