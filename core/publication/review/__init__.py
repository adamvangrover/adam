"""Review layer: expert-desk critique, adversarial red team and clean-room judge for each edition.

Offline by default (``config/publication_review.yaml`` → ``llm.enabled: false``): every verdict comes from
deterministic critics. Prompts for the optional LLM path live in ``prompt_library/publication_review/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from core.publication.review.critics import fragile_targets, run_critics
from core.publication.review.judge import judge_edition
from core.publication.review.red_team import red_team_review
from core.schemas.publication import Edition

REPO_ROOT = Path(__file__).resolve().parents[3]
REVIEW_CONFIG = "config/publication_review.yaml"
REVIEW_VERSION = "1.0.0"


def load_review_config(path: str = REVIEW_CONFIG) -> Dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / path).read_text())


def _as_pair(item: Union[Edition, Dict[str, Any]]) -> tuple[Dict[str, Any], Edition]:
    if isinstance(item, Edition):
        return item.to_record(), item
    return item, Edition.model_validate(item)


def review_flags(record: Union[Edition, Dict[str, Any]], prior: Union[Edition, Dict[str, Any], None] = None,
                 cfg: Optional[Dict[str, Any]] = None) -> list:
    """All deterministic flags (critics + red-team fragility); used by the eval dataset."""
    cfg = cfg or load_review_config()
    raw, edition = _as_pair(record)
    prior_ed = _as_pair(prior)[1] if prior is not None else None
    return run_critics(raw, edition, prior_ed, cfg) + fragile_targets(edition, cfg)


def _desk(flags: list) -> Dict[str, Any]:
    sev = {f["severity"] for f in flags}
    verdict = "block" if "error" in sev else "revise" if "warning" in sev else "publish"
    return {"persona": "expert_desk", "prompt": "prompt_library/publication_review/expert_desk.md", "verdict": verdict,
            "concerns": [{"hook": f["hook"], "issue": f["message"], "severity": f["severity"]}
                         for f in flags if f["code"] != "assumption_fragile"]}


def review_edition(record: Union[Edition, Dict[str, Any]], prior: Union[Edition, Dict[str, Any], None] = None,
                   cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_review_config()
    raw, edition = _as_pair(record)
    prior_ed = _as_pair(prior)[1] if prior is not None else None
    findings = run_critics(raw, edition, prior_ed, cfg)
    red = red_team_review(edition, findings, cfg)
    flags = findings + [c for c in red["challenges"] if c["code"] == "assumption_fragile"]
    judge = judge_edition(edition, cfg)
    bull = next((s for s in edition.scenarios if s.name == "bull"), None)
    bear = next((s for s in edition.scenarios if s.name == "bear"), None)
    return {
        "version": REVIEW_VERSION, "mode": "llm" if cfg["llm"]["enabled"] else "offline", "config": REVIEW_CONFIG,
        "flags": flags, "desk": _desk(flags), "red_team": red, "judge": judge, "score": judge["overall_score"],
        "panel": {
            "bull": {"narrative": bull.narrative if bull else "", "targets": bull.targets if bull else {},
                     "price_targets": {t.ticker: t.bull for t in edition.price_targets}},
            "bear": {"narrative": bear.narrative if bear else "", "targets": bear.targets if bear else {},
                     "price_targets": {t.ticker: t.bear for t in edition.price_targets}},
            "red_team": {"summary": red["report"].splitlines()[2] if red["report"] else "",
                         "challenges": [c["message"] for c in red["challenges"]]},
        },
    }
