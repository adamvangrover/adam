"""Stage 3: System 2 Legal & Financial Covenant Solver (Catala + Z3 + OpenFisca)."""
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
