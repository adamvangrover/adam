"""System 1 Triage & Extraction (jev-rs + outlines + edgartools + langextract pattern)."""
from __future__ import annotations

import re

from src.domains.neuro_symbolic_ports import (
    EpistemicTier,
    GroundedSpan,
    JEV_ADAPTER,
    JevSystem1Judgment,
)

import importlib.util

HAS_LLGUIDANCE = importlib.util.find_spec("llguidance") is not None


def extract_grounded_spans(
    source_doc_id: str,
    raw_text: str,
    patterns: dict[str, EpistemicTier] | None = None,
) -> list[GroundedSpan]:
    """
    Extracts verbatim text spans with exact character offsets and epistemic tier tagging.
    """
    spans: list[GroundedSpan] = []
    if patterns is None:
        patterns = {
            r"\b(?:EBITDA|EBITDAR|DSCR|CADS)\b": "VERIFIED",
            r"\b(?:\d+(?:\.\d+)?%|\$\d+(?:\.\d+)?(?:\s*(?:million|billion|M|B))?)\b": "CALCULATED",
            r"\b(?:covenant|default|breach|remedy|cure)\b": "VERIFIED",
        }

    for pattern_str, tier in patterns.items():
        for m in re.finditer(pattern_str, raw_text, flags=re.IGNORECASE):
            spans.append(
                GroundedSpan(
                    source_doc_id=source_doc_id,
                    char_start=m.start(),
                    char_end=m.end(),
                    verbatim_text=m.group(0),
                    tier=tier,
                )
            )
    return spans


def triage_inference(
    intent_choice: str,
    conviction_score: float,
    grounded_spans: list[GroundedSpan],
    metadata: dict[str, str] | None = None,
) -> JevSystem1Judgment:
    """
    Constructs a validated JevSystem1Judgment and enforces tripwire thresholds.
    """
    judgment = JevSystem1Judgment(
        intent_choice=intent_choice,  # type: ignore[arg-type]
        conviction_score=conviction_score,
        next_step_metadata=metadata or {},
        grounded_spans=tuple(grounded_spans),
    )
    return judgment.enforce_tripwire(min_conviction=0.85)


def get_jev_json_schema() -> dict:
    """Returns the JSON schema corresponding to JevSystem1Judgment for token masking."""
    return JevSystem1Judgment.model_json_schema()


def build_llguidance_grammar_spec() -> dict:
    """
    Builds structured llguidance grammar specification for zero-overhead token masking.
    Returns grammar spec dictionary compatible with llguidance LLM runners.
    """
    schema = get_jev_json_schema()
    return {
        "type": "json_schema",
        "schema": schema,
        "llguidance_available": HAS_LLGUIDANCE,
    }


__all__ = [
    "EpistemicTier",
    "GroundedSpan",
    "JevSystem1Judgment",
    "JEV_ADAPTER",
    "extract_grounded_spans",
    "triage_inference",
    "get_jev_json_schema",
    "build_llguidance_grammar_spec",
]
