import hashlib
import json
from typing import Dict, Any


def run(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Skill 2 — Quantitative Risk Engine Interoperability
    Executes approved PD/LGD/EAD quantitative challenger calculations ($EL = PD \times LGD \times EAD$),
    compares against incumbent results under tolerance limits, and emits non-authoritative proposals.
    """
    correlation_id = input_data.get("correlation_id", "unknown-correlation-id")
    obligor_id = input_data.get("obligor_id", "UNKNOWN_OBLIGOR")

    pd_fraction = float(input_data.get("pd_fraction", 0.0))
    lgd_fraction = float(input_data.get("lgd_fraction", 0.0))
    ead_amount = float(input_data.get("ead_amount", 0.0))
    fccr_ratio = float(input_data.get("fccr_ratio", 0.0))
    incumbent_el = float(input_data.get("incumbent_el_amount", 0.0))
    tolerance_threshold = float(input_data.get("tolerance_threshold", 100.0))

    # Unit & range validation
    if not (0.0 <= pd_fraction <= 1.0):
        raise ValueError(f"pd_fraction must be in range [0, 1], got {pd_fraction}. Check if percentage was passed!")
    if not (0.0 <= lgd_fraction <= 1.0):
        raise ValueError(f"lgd_fraction must be in range [0, 1], got {lgd_fraction}.")
    if ead_amount < 0.0:
        raise ValueError(f"ead_amount cannot be negative, got {ead_amount}.")

    # Deterministic Arithmetic: Expected Loss = PD * LGD * EAD
    challenger_el = pd_fraction * lgd_fraction * ead_amount
    variance_amount = abs(challenger_el - incumbent_el)
    tolerance_exceeded = variance_amount > tolerance_threshold

    return {
        "skill_id": "pd_model_challenger",
        "status": "COMPLETED",
        "obligor_id": obligor_id,
        "challenger_el_amount": round(challenger_el, 4),
        "incumbent_el_amount": round(incumbent_el, 4),
        "variance_amount": round(variance_amount, 4),
        "tolerance_exceeded": tolerance_exceeded,
        "correlation_id": correlation_id,
        "authority_level": "proposal_only"
    }
