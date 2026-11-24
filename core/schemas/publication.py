"""Unified publication schema: one ``Edition`` record per newsletter / report edition.

Data first, HTML second: every edition is validated here, written as JSON (+MD), then rendered by
``core/publication/render.py``. HTML is never the source of truth.

Values not computed from a committed dataset carry ``_estimated: true`` (``docs/PROVENANCE.md``);
``Edition.estimated`` is true when any part of the edition is estimated, and renderers then show the
"illustrative scenario — not investment advice" label.
"""
from __future__ import annotations

import math
from datetime import date as Date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Family = Literal[
    "market_mayhem", "daily_briefing", "market_pulse", "house_view",
    "equity_research", "sector_outlook", "credit_memo", "deep_dive",
]
FAMILIES: tuple[str, ...] = Family.__args__  # type: ignore[attr-defined]
SCHEMA_VERSION = "1.0.0"
ESTIMATED_LABEL = "Illustrative scenario — not investment advice"


class _Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class _Estimable(_Model):
    estimated: bool = Field(False, alias="_estimated", description="True when not computed from a committed dataset.")


class ProvenanceRecord(_Model):
    """Who produced a record, from what, when, with which code version."""

    generator: str = Field(..., description="Repo-relative script or module that produced the record.")
    inputs: List[str] = Field(default_factory=list, description="Repo-relative input paths / dataset ids.")
    timestamp: str = Field(..., description="ISO-8601 generation time (deterministic for backfills).")
    version: str = Field(SCHEMA_VERSION, description="Generator / schema version.")
    content_hash: Optional[str] = Field(None, description="SHA-256 of the canonical inputs, when known.")


class Source(_Model):
    label: str
    ref: str = Field(..., description="Repo path or URL of the source.")
    as_of: Optional[str] = None


class Section(_Model):
    hook: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]*$", description="data-adam-hook id for the section.")
    title: str
    body: str = Field("", description="Plain-text / Markdown body; renderers escape it.")
    bullets: List[str] = Field(default_factory=list)


class SnapshotItem(_Estimable):
    name: str
    value: float
    unit: str = ""
    change: Optional[float] = None
    change_unit: str = ""


class Call(_Estimable):
    """A directional view (asset, sector or ticker)."""

    subject: str
    stance: Literal["overweight", "neutral", "underweight", "long", "short", "hedge", "watch"]
    conviction: float = Field(..., ge=0.0, le=1.0)
    rationale: str = ""


class Risk(_Estimable):
    name: str
    probability: float = Field(..., ge=0.0, le=1.0)
    impact: Literal["low", "medium", "high", "severe"]
    description: str = ""


class Scenario(_Estimable):
    name: Literal["base", "bull", "bear"]
    probability: float = Field(..., ge=0.0, le=1.0)
    narrative: str = ""
    targets: Dict[str, float] = Field(default_factory=dict, description="Key levels under this scenario.")


class PriceTarget(_Estimable):
    """Deterministic price target; the engine output, never LLM-written numbers."""

    ticker: str = Field(..., pattern=r"^[A-Z][A-Z0-9.\-]{0,9}$")
    method: Literal["dcf", "multiples", "scenario_weighted"]
    base: float
    bull: float
    bear: float
    weights: Dict[Literal["base", "bull", "bear"], float] = Field(
        default_factory=lambda: {"base": 0.5, "bull": 0.25, "bear": 0.25})
    horizon_months: int = Field(12, ge=1, le=60)
    inputs_ref: str = Field(..., description="Repo path of the committed inputs the target was computed from.")
    reference_price: Optional[float] = None

    @model_validator(mode="after")
    def _check(self) -> "PriceTarget":
        if not self.bear <= self.base <= self.bull:
            raise ValueError("price targets must satisfy bear <= base <= bull")
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError("scenario weights must sum to 1")
        return self

    @property
    def weighted(self) -> float:
        return round(math.fsum(getattr(self, k) * w for k, w in self.weights.items()), 2)


class Edition(_Model):
    family: Family
    date: Date
    title: str = Field(..., min_length=1)
    summary: str = ""
    sections: List[Section] = Field(default_factory=list)
    market_snapshot: List[SnapshotItem] = Field(default_factory=list)
    calls: List[Call] = Field(default_factory=list)
    price_targets: List[PriceTarget] = Field(default_factory=list)
    risks: List[Risk] = Field(default_factory=list)
    scenarios: List[Scenario] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    provenance_trace: List[ProvenanceRecord] = Field(..., min_length=1)
    review: Dict[str, Any] = Field(default_factory=dict)
    estimated: bool = Field(False, alias="_estimated")
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _propagate_estimated(self) -> "Edition":
        parts = [*self.market_snapshot, *self.calls, *self.price_targets, *self.risks, *self.scenarios]
        if any(p.estimated for p in parts):
            self.estimated = True
        hooks = [s.hook for s in self.sections]
        if len(hooks) != len(set(hooks)):
            raise ValueError("section hooks must be unique")
        return self

    @property
    def edition_id(self) -> str:
        return f"{self.family}/{self.date.isoformat()}"

    def to_record(self) -> Dict[str, Any]:
        """JSON-ready dict using the on-disk field names (``_estimated``)."""
        return self.model_dump(mode="json", by_alias=True)
