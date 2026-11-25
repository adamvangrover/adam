"""Deterministic, offline rule critics for publication editions (thresholds: config/publication_review.yaml).

Critics read the raw on-disk record (so an edition-level ``_estimated: false`` that the schema would
silently correct is still caught) plus the validated ``Edition`` and, optionally, the prior edition.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from core.schemas.publication import Edition

_PARTS = ("market_snapshot", "calls", "price_targets", "risks", "scenarios")
_OPPOSITE = {("overweight", "underweight"), ("underweight", "overweight"), ("long", "short"), ("short", "long")}


def _flag(cfg: Dict[str, Any], code: str, hook: str, message: str) -> Dict[str, str]:
    return {"code": code, "severity": cfg["severity"][code], "hook": hook, "message": message}


def _sources_and_labels(record, edition, cfg) -> List[Dict[str, str]]:
    out = []
    if not edition.sources:
        out.append(_flag(cfg, "missing_sources", "provenance", "Edition cites no sources."))
    parts_estimated = any(item.get("_estimated") for p in _PARTS for item in record.get(p, []))
    if parts_estimated and not record.get("_estimated"):
        out.append(_flag(cfg, "estimated_unlabelled", "estimated-label",
                         "Estimated figures present but the edition is not flagged/labelled as estimated."))
    prefixes = tuple(cfg["critics"]["synthetic_source_prefixes"])
    only_synthetic = all(s.ref.startswith(prefixes) for s in edition.sources)
    unflagged = [item.get("name") or item.get("subject") or item.get("ticker")
                 for p in ("market_snapshot", "price_targets") for item in record.get(p, []) if not item.get("_estimated")]
    if unflagged and only_synthetic:
        out.append(_flag(cfg, "unsourced_value", "market-snapshot",
                         f"Values without `_estimated` and without a non-synthetic source: {', '.join(map(str, unflagged))}."))
    return out


def _targets(edition, prior, cfg) -> List[Dict[str, str]]:
    c, out = cfg["critics"], []
    prior_base = {t.ticker: t.base for t in (prior.price_targets if prior else [])}
    for t in edition.price_targets:
        if t.reference_price:
            gap = t.weighted / t.reference_price - 1
            if abs(gap) > c["max_target_vs_reference"]:
                out.append(_flag(cfg, "target_out_of_range", "price-targets",
                                 f"{t.ticker} weighted target {t.weighted:,.2f} is {gap:+.0%} vs reference {t.reference_price:,.2f}."))
        if t.ticker in prior_base and prior_base[t.ticker]:
            jump = t.base / prior_base[t.ticker] - 1
            if abs(jump) > c["max_target_jump_vs_prior"]:
                out.append(_flag(cfg, "target_jump", "price-targets",
                                 f"{t.ticker} base target moved {jump:+.0%} vs prior edition."))
    return out


def _calls(edition, prior, cfg) -> List[Dict[str, str]]:
    c, out = cfg["critics"], []
    prior_calls = {x.subject: x for x in (prior.calls if prior else [])}
    for call in edition.calls:
        if call.conviction >= c["unsupported_min_conviction"] and not call.rationale.strip():
            out.append(_flag(cfg, "unsupported_call", "calls", f"High-conviction call on {call.subject} has no rationale."))
        p = prior_calls.get(call.subject)
        if (p and (p.stance, call.stance) in _OPPOSITE
                and min(p.conviction, call.conviction) >= c["contradiction_min_conviction"]):
            out.append(_flag(cfg, "contradicts_prior", "calls",
                             f"{call.subject}: {p.stance} -> {call.stance} at high conviction with no stated catalyst."))
    return out


def _freshness(edition: Edition, cfg) -> List[Dict[str, str]]:
    dated = [date.fromisoformat(s.as_of[:10]) for s in edition.sources if s.as_of]
    if dated and (edition.date - max(dated)).days > cfg["critics"]["stale_after_days"]:
        return [_flag(cfg, "stale_data", "provenance",
                      f"Newest source is {(edition.date - max(dated)).days} days older than the edition.")]
    return []


def _scenarios(edition: Edition, cfg) -> List[Dict[str, str]]:
    if edition.scenarios and abs(sum(s.probability for s in edition.scenarios) - 1) > cfg["critics"]["probability_tolerance"]:
        return [_flag(cfg, "scenario_probabilities", "scenarios", "Scenario probabilities do not sum to 1.")]
    return []


def fragile_targets(edition: Edition, cfg) -> List[Dict[str, str]]:
    """Red-team challenge: what if the bear case were the base case?"""
    limit = cfg["red_team"]["bear_as_base_max_drawdown"]
    return [_flag(cfg, "assumption_fragile", "price-targets",
                  f"{t.ticker}: bear-as-base implies {t.bear / t.weighted - 1:+.0%} vs the weighted target.")
            for t in edition.price_targets if t.weighted and t.bear / t.weighted - 1 < -limit]


def run_critics(record: Dict[str, Any], edition: Edition, prior: Optional[Edition], cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    return [*_sources_and_labels(record, edition, cfg), *_targets(edition, prior, cfg), *_calls(edition, prior, cfg),
            *_freshness(edition, cfg), *_scenarios(edition, cfg)]
