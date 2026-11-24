"""Deterministic Base/Bull/Bear price targets for the publication coverage universe.

DCF names run ``core.engine.valuation_utils.calculate_dcf`` once per scenario; banks use a P/E multiple
band (``calculate_multiples`` is not applicable without EV/EBITDA). All assumptions come from
``config/valuation/coverage_2026.yaml``; the risk-free rate comes from the committed scenario path.
No LLM touches these numbers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from core.engine.valuation_utils import calculate_dcf
from core.schemas.publication import PriceTarget

REPO_ROOT = Path(__file__).resolve().parents[2]
COVERAGE_CONFIG = "config/valuation/coverage_2026.yaml"


def load_coverage(path: str = COVERAGE_CONFIG) -> Dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / path).read_text())


def _dcf_targets(spec: Dict[str, Any], cfg: Dict[str, Any], rfr: float) -> Dict[str, float]:
    out = {}
    for name, sc in cfg["dcf_scenarios"].items():
        scenario = {
            "growth_rate": spec["growth_rate"] + sc["growth_delta"],
            "beta": spec["beta"] + sc["beta_delta"],
            "market_risk_premium": sc["market_risk_premium"],
            "risk_free_rate": rfr,
        }
        out[name] = calculate_dcf(spec, risk_free_rate=rfr, scenario=scenario)["intrinsic_share_price"]
    return out


def _multiple_targets(spec: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, float]:
    return {name: round(spec["eps"] * spec["pe"] * k, 2) for name, k in cfg["multiples_scenarios"].items()}


def compute_targets(rfr_pct: float, cfg: Dict[str, Any] | None = None) -> List[PriceTarget]:
    """Price targets for every covered ticker given the risk-free rate (percent, e.g. 4.8)."""
    cfg = cfg or load_coverage()
    targets = []
    for ticker, spec in cfg["coverage"].items():
        if spec["method"] == "dcf":
            vals, method = _dcf_targets(spec, cfg, rfr_pct / 100.0), "dcf"
        else:
            vals, method = _multiple_targets(spec, cfg), "multiples"
        bear, base, bull = sorted((vals["bear"], vals["base"], vals["bull"]))
        targets.append(PriceTarget(
            ticker=ticker, method=method, base=round(base, 2), bull=round(bull, 2), bear=round(bear, 2),
            weights=cfg["weights"], horizon_months=cfg["horizon_months"],
            inputs_ref=COVERAGE_CONFIG, reference_price=spec["reference_price"], estimated=True,
        ))
    return targets
