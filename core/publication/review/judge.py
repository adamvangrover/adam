"""Weighted ``CleanRoomLLMJudge`` score for an edition's narrative against its own structured data."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

from core.schemas.publication import Edition
from evals.graders.llm_judge import CleanRoomLLMJudge


def headline(edition: Edition) -> str:
    """The structured fact the narrative must state: the largest normalised mover in the snapshot."""
    movers = [s for s in edition.market_snapshot if s.change is not None]
    if not movers:
        return edition.title
    top = max(movers, key=lambda s: abs(s.change) / (1.0 if s.change_unit == "%" else 10.0))
    return f"{top.value:,}"


def judge_edition(edition: Edition, cfg: Dict[str, Any], llm_callable: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    j = cfg["judge"]
    judge = CleanRoomLLMJudge(llm_callable=llm_callable if cfg["llm"]["enabled"] else None,
                              weights=dict(j["weights"]), pass_threshold=j["pass_threshold"])
    narrative = " ".join([edition.summary, *(s.body for s in edition.sections),
                          "Sources: " + "; ".join(s.ref for s in edition.sources)])
    context = " ".join([*(f"{s.name} {s.value:,} {s.unit} {s.change} {s.change_unit}" for s in edition.market_snapshot),
                        *(f"{c.subject} {c.stance} {c.rationale}" for c in edition.calls),
                        *(f"{t.ticker} {t.weighted:,}" for t in edition.price_targets),
                        *(f"{s.label} {s.ref}" for s in edition.sources),
                        json.dumps([s.model_dump(mode="json") for s in edition.scenarios])])
    result = judge.evaluate(question=edition.title, agent_answer=narrative, golden_answer=headline(edition), context=context)
    return {"overall_score": result.overall_score, "passed": result.passed, "verdict": result.verdict,
            "weights": dict(j["weights"]),
            "dimensions": [{"name": d.name, "score": round(d.score, 4), "weight": d.weight, "reasoning": d.reasoning}
                           for d in result.dimensions]}
