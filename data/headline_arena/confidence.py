"""
Confidence mapping: granular internal classifier with fixed fallback.

No uplift: confidence starts at 0.50 (platform floor) and rises only with the
model's *margin* over the runner-up outcome, then is adjusted by dead-zone
geometry and the asset's settled track record (calibration.json from learn.py).

Modes (env HA_CONF_MODE):
  auto  (default) calibrated when calibration.json has history for the asset,
                  otherwise fixed
  fixed           tier table only — deterministic, for manual runs
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

CALIB_PATH = Path(__file__).parent.parent / "data" / "memory" / "calibration.json"

FLOOR = 0.50
DEFAULT_CAP = 0.60
MIN_HISTORY = 2  # settled predictions needed before calibration is trusted

# Margin = p(chosen) - p(runner-up). Upper bounds are exclusive.
TIERS = (("TIE", 0.03), ("LEAN", 0.07), ("MODERATE", 0.12), ("STRONG", 1.01))
FIXED_TABLE = {"TIE": 0.50, "LEAN": 0.53, "MODERATE": 0.57, "STRONG": 0.62}

DZ_PENALTY = 0.03       # directional call whose median move sits inside the band
NEUTRAL_EDGE_PENALTY = 0.03  # neutral call whose median move sits near the band edge


@dataclass
class ConfidenceDecision:
    confidence: float
    tier: str
    mode: str
    margin: float
    reasons: list

    def summary(self) -> str:
        return f"{self.tier}/{self.mode} margin={self.margin:.3f}: " + "; ".join(self.reasons)


def _tier(margin: float) -> str:
    for name, upper in TIERS:
        if margin < upper:
            return name
    return "STRONG"


NEUTRAL_MARGIN = 0.0  # production default; tuned via backtest


def choose_direction(probs: Dict[str, float], neutral_margin: float = None) -> str:
    """Argmax, except call neutral when p(neutral) is within `neutral_margin` of the top."""
    m = NEUTRAL_MARGIN if neutral_margin is None else neutral_margin
    top_dir = "bull" if probs["bull"] > probs["bear"] else "bear"
    if probs["neutral"] >= probs[top_dir] - m:
        return "neutral"
    return "bullish" if top_dir == "bull" else "bearish"


def _asset_calib(ticker: str) -> Optional[Dict]:
    try:
        a = json.loads(CALIB_PATH.read_text()).get("assets", {}).get(ticker)
    except (OSError, ValueError):
        return None
    return a if a and a.get("n", 0) >= MIN_HISTORY else None


def map_confidence(
    ticker: str,
    direction: str,
    probs: Dict[str, float],          # keys: bull, neutral, bear (sum ~1)
    exp_move_pct: float,              # median simulated move, %
    dead_zone_pct: float,
    mode: Optional[str] = None,
    params: Optional[Dict] = None,
    calib_override: Optional[Dict] = None,
) -> ConfidenceDecision:
    """
    params (all optional; defaults = production):
      band_rule: 'absolute' (flat penalty if median inside band) |
                 'relative' (penalty scales with 1 - |move|/band; zero at band edge)
      dz_penalty: float, rel_max: float (reliability multiplier ceiling)
    calib_override: per-asset stats dict (walk-forward backtests); bypasses file.
    """
    P = {"band_rule": "absolute", "dz_penalty": DZ_PENALTY, "rel_max": 1.0, **(params or {})}  # rel_max 1.0 per backtest
    key = {"bullish": "bull", "bearish": "bear", "neutral": "neutral"}[direction]
    p_dir = probs[key]
    runner_up = max(v for k, v in probs.items() if k != key)
    margin = max(0.0, p_dir - runner_up)
    tier = _tier(margin)
    reasons = [f"p={p_dir:.3f} vs runner-up {runner_up:.3f}"]

    mode = (mode or os.environ.get("HA_CONF_MODE", "auto")).lower()
    calib = None
    if mode != "fixed":
        calib = calib_override if calib_override is not None else _asset_calib(ticker)
        if calib is not None and calib.get("n", 0) < MIN_HISTORY:
            calib = None
    if mode == "fixed" or calib is None:
        used = "fixed"
        conf = FIXED_TABLE[tier]
        if mode != "fixed":
            reasons.append("no calibration history -> fixed table")
    else:
        used = "calibrated"
        # Linear in margin from the floor; STRONG margins (~0.15) reach ~0.65.
        conf = FLOOR + margin
        cap = float(calib.get("suggested_conf_cap", DEFAULT_CAP))
        # Reliability: shrink the above-floor part by how over/under-confident
        # this asset has been (overconfidence > 0 -> shrink).
        over = float(calib.get("overconfidence", 0.0))
        rel = min(P["rel_max"], max(0.5, 1.0 - over))
        conf = FLOOR + (conf - FLOOR) * rel
        conf = min(conf, cap)
        reasons.append(f"cap={cap:.2f} reliability x{rel:.2f} (n={calib['n']}, hit={calib['hit_rate']:.0%})")

    # Dead-zone geometry
    inside = abs(exp_move_pct) <= dead_zone_pct
    if direction != "neutral" and inside:
        if P["band_rule"] == "relative" and dead_zone_pct > 0:
            pen = P["dz_penalty"] * (1.0 - abs(exp_move_pct) / dead_zone_pct)
        else:
            pen = P["dz_penalty"]
        conf -= pen
        reasons.append(f"median move {exp_move_pct:+.2f}% inside ±{dead_zone_pct:.2f}% band (-{pen:.3f})")
    if direction == "neutral" and abs(exp_move_pct) > 0.6 * dead_zone_pct:
        conf -= NEUTRAL_EDGE_PENALTY
        reasons.append(f"neutral but median move {exp_move_pct:+.2f}% near band edge")

    conf = round(max(FLOOR, conf), 2)
    return ConfidenceDecision(conf, tier, used, round(margin, 4), reasons)
