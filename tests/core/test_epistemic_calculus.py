import pytest
from src.core.epistemic_calculus import EpistemicCalculus
from src.schemas.epistemic_types import EpistemicStatus

def test_epistemic_calculus_unknown():
    calc = EpistemicCalculus(e_min=0.5, tau_ood=0.8, tau_w=0.3)
    # Insufficient evidence on critical path -> UNKNOWN
    assert calc.evaluate_state(0.4, 0.1, 0.1, True, True) == EpistemicStatus.UNKNOWN
    # Conflicted without resolution policy -> UNKNOWN
    assert calc.evaluate_state(0.9, 0.1, 0.5, False, False) == EpistemicStatus.UNKNOWN

def test_epistemic_calculus_insufficient():
    calc = EpistemicCalculus(e_min=0.5, tau_ood=0.8, tau_w=0.3)
    assert calc.evaluate_state(0.4, 0.1, 0.1, True, False) == EpistemicStatus.INSUFFICIENT_EVIDENCE
