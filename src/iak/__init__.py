try:
    from .adam_iak_primitives import (
        DtypeSafeMaskedEntmaxSoftmax,
        UncertaintyGatedCRFSlotDecoder,
        DeterministicPolicyEpistemicGate,
        bisect_threshold_entmax15,
        topk_entmax15,
    )
    _PRIMITIVES_AVAILABLE = True
except ImportError:
    _PRIMITIVES_AVAILABLE = False

from .s1_triage_extract import (
    EpistemicTier,
    GroundedSpan,
    JevSystem1Judgment,
    JEV_ADAPTER,
    extract_grounded_spans,
    triage_inference,
)

__all__ = [
    "EpistemicTier",
    "GroundedSpan",
    "JevSystem1Judgment",
    "JEV_ADAPTER",
    "extract_grounded_spans",
    "triage_inference",
]

if _PRIMITIVES_AVAILABLE:
    __all__.extend([
        "DtypeSafeMaskedEntmaxSoftmax",
        "UncertaintyGatedCRFSlotDecoder",
        "DeterministicPolicyEpistemicGate",
        "bisect_threshold_entmax15",
        "topk_entmax15",
    ])
