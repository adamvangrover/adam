"""Build schema-valid ``Edition`` records from the committed synthetic path (Phase 2 backfill).

Numbers come only from ``data/scenarios/2026H2_path.json`` and the deterministic valuation engine;
rules, thresholds and phrasing come from ``config/publications.yaml``. Every record is ``_estimated``.
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.publication.valuation import COVERAGE_CONFIG, compute_targets
from core.schemas.publication import Call, Edition, ProvenanceRecord, Risk, Scenario, Section, SnapshotItem, Source

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH_FILE = "data/scenarios/2026H2_path.json"
PUB_CONFIG = "config/publications.yaml"
GENERATOR = "scripts/publications/backfill_editions.py"
VERSION = "1.0.0"


class ScenarioPath:
    """Lookup helper over the committed path JSON."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.dates = [date.fromisoformat(d) for d in data["dates"]]
        self._idx = {d: i for i, d in enumerate(self.dates)}
        self.series = data["series"]

    @classmethod
    def load(cls, path: str = PATH_FILE) -> "ScenarioPath":
        return cls(json.loads((REPO_ROOT / path).read_text()))

    def value(self, name: str, d: date) -> float:
        return self.series[name]["values"][self._idx[d]]

    def change(self, name: str, d: date, lookback: int, mode: str) -> float:
        i = self._idx[d]
        now, then = self.value(name, d), self.series[name]["values"][max(0, i - lookback)]
        return round((now / then - 1) * 100, 2) if mode == "log" else round((now - then) * 100, 1)

    def vol(self, name: str, d: date, window: int = 21) -> float:
        i = self._idx[d]
        vals = self.series[name]["values"][max(0, i - window): i + 1]
        rets = [math.log(b / a) for a, b in zip(vals, vals[1:])]
        if len(rets) < 2:
            return 0.01
        mean = sum(rets) / len(rets)
        return math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1))


def load_config(path: str = PUB_CONFIG) -> Dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / path).read_text())


def _modes() -> Dict[str, str]:
    cfg = yaml.safe_load((REPO_ROOT / "config/scenarios/2026H2_path.yaml").read_text())
    return {k: v["mode"] for k, v in cfg["series"].items()}


def edition_dates(family: str, cfg: Dict[str, Any], path: ScenarioPath) -> List[date]:
    fam = cfg["families"][family]
    lo, hi = (date.fromisoformat(cfg["window"][k]) for k in ("start", "end"))
    days = [d for d in path.dates if lo <= d <= hi]
    if fam["cadence"] == "daily":
        return days
    if fam["cadence"] == "weekly":
        return [d for d in days if d.weekday() == fam["weekday"]]
    months = {1, 4, 7, 10} if fam["cadence"] == "quarterly" else set(range(1, 13))
    firsts: Dict[tuple, date] = {}
    for d in days:
        if d.month in months:
            firsts.setdefault((d.year, d.month), d)
    return sorted(firsts.values())


def regime(cfg: Dict[str, Any], vix: float) -> str:
    return next(r["name"] for r in cfg["regimes"] if vix >= r["vix_min"])


def _fmt_chg(chg: float, mode: str) -> str:
    return f"{chg:+.2f}%" if mode == "log" else f"{chg:+.1f}bp"


class EditionBuilder:
    def __init__(self, path: ScenarioPath, cfg: Dict[str, Any], mm_refs: Optional[Dict[date, str]] = None):
        self.path, self.cfg, self.modes, self.mm_refs = path, cfg, _modes(), mm_refs or {}

    def _snapshot(self, d: date, lookback: int) -> List[SnapshotItem]:
        items = []
        for name in self.cfg["snapshot_series"]:
            mode, s = self.modes[name], self.path.series[name]
            items.append(SnapshotItem(name=s["label"], value=self.path.value(name, d), unit=s["unit"],
                                      change=self.path.change(name, d, lookback, mode),
                                      change_unit=self.cfg["change_units"][mode], estimated=True))
        return items

    def _lead(self, d: date, lookback: int, cadence: str) -> tuple[str, str]:
        ph = self.cfg["phrases"]
        moves = []
        for name in self.cfg["snapshot_series"]:
            mode = self.modes[name]
            chg = self.path.change(name, d, lookback, mode)
            score = abs(chg) / (1.0 if mode == "log" else 10.0)
            verb = ph["flat"] if score < 0.1 else ph["up"] if chg > 0 else ph["down"]
            text = ph["move"].format(label=self.path.series[name]["label"], verb=verb,
                                     chg=_fmt_chg(chg, mode), value=f"{self.path.value(name, d):,}")
            moves.append((score, text, text.rsplit(' to ', 1)[0]))
        moves.sort(key=lambda m: -m[0])
        body = ph["lead_section"].format(window=ph["window"][cadence], moves="; ".join(m[1] for m in moves[:3]))
        return body, moves[0][2]

    def _calls(self, d: date, lookback: int) -> List[Call]:
        cv, calls = self.cfg["conviction"], []
        for rule in self.cfg["call_rules"]:
            chg = self.path.change(rule["series"], d, lookback, self.modes[rule["series"]])
            if chg >= rule["up"]:
                stance, mult = rule["if_up"], chg / rule["up"]
            elif chg <= rule["down"]:
                stance, mult = rule["if_down"], chg / rule["down"]
            else:
                stance, mult = "neutral", 0.0
            conviction = min(cv["max"], cv["base"] + cv["per_threshold"] * mult)
            calls.append(Call(subject=rule["subject"], stance=stance, conviction=round(conviction, 2),
                              rationale=f"{rule['series']} {_fmt_chg(chg, self.modes[rule['series']])} over lookback",
                              estimated=True))
        return calls

    def _scenarios(self, d: date) -> List[Scenario]:
        sc = self.cfg["scenarios"]
        spx, vol = self.path.value("spx", d), self.path.vol("spx", d)
        out = []
        for name in ("base", "bull", "bear"):
            target = spx * (1 + sc[name]["k"] * vol * math.sqrt(sc["horizon_days"]))
            out.append(Scenario(name=name, probability=sc[name]["probability"], narrative=sc["narrative"][name],
                                targets={"S&P 500": round(target, 0)}, estimated=True))
        return out

    def _risks(self, d: date) -> List[Risk]:
        scale = min(1.6, max(0.5, self.path.value("vix", d) / 20.0))
        return [Risk(name=r["name"], probability=round(min(0.95, r["base"] * scale), 2), impact=r["impact"],
                     description=r["description"], estimated=True) for r in self.cfg["risks"]]

    def _sector_calls(self, d: date, lookback: int) -> List[Call]:
        calls = []
        for sector, spec in self.cfg["sectors"].items():
            chg = self.path.change(spec["series"], d, lookback, self.modes[spec["series"]]) * spec["sign"]
            stance = "overweight" if chg > 0 else "underweight" if chg < 0 else "neutral"
            calls.append(Call(subject=sector, stance=stance, conviction=0.5, estimated=True,
                              rationale=f"driver {spec['series']} signed change {chg:+.2f}"))
        return calls

    def _pt_calls(self, targets) -> List[Call]:
        th = self.cfg["pt_calls"]
        out = []
        for t in targets:
            upside = t.weighted / t.reference_price - 1
            stance = ("overweight" if upside >= th["overweight_upside"] else
                      "underweight" if upside <= th["underweight_downside"] else "neutral")
            out.append(Call(subject=t.ticker, stance=stance, conviction=0.5, estimated=True,
                            rationale=f"weighted target {t.weighted:,.2f} vs reference {t.reference_price:,.2f} ({upside:+.1%})"))
        return out

    def build(self, family: str, d: date) -> Edition:
        fam, ph = self.cfg["families"][family], self.cfg["phrases"]
        cadence = fam["cadence"]
        lookback = self.cfg["lookback_days"][cadence]
        vix = self.path.value("vix", d)
        reg = regime(self.cfg, vix)
        lead_body, lead_label = self._lead(d, lookback, cadence)
        sections = [Section(hook="lead", title="Lead", body=lead_body),
                    Section(hook="regime", title="Regime", body=ph["regime_section"].format(vix=vix, regime=reg)),
                    Section(hook="disclaimer", title="Data note", body=ph["disclaimer"])]
        inputs = [PATH_FILE, PUB_CONFIG]
        sources = [Source(label="Synthetic 2026H2 path", ref=PATH_FILE, as_of=d.isoformat()),
                   Source(label="Publication rules", ref=PUB_CONFIG)]
        kw: Dict[str, Any] = {"market_snapshot": self._snapshot(d, lookback)}
        if family in ("market_mayhem", "daily_briefing", "market_pulse", "house_view"):
            kw["calls"] = self._calls(d, lookback)
        if family in ("market_mayhem", "house_view"):
            kw["scenarios"], kw["risks"] = self._scenarios(d), self._risks(d)
        if family == "sector_outlook":
            kw["calls"], kw["risks"] = self._sector_calls(d, lookback), self._risks(d)
        if family == "equity_research":
            rfr = self.path.value("ust10y", d)
            targets = compute_targets(rfr)
            kw["price_targets"], kw["calls"] = targets, self._pt_calls(targets)
            sections.insert(1, Section(hook="valuation", title="Valuation method", body=ph["pt_section"].format(rfr=rfr)))
            inputs.append(COVERAGE_CONFIG)
            sources.append(Source(label="Coverage assumptions", ref=COVERAGE_CONFIG))
        ref = next((r for rd, r in self.mm_refs.items() if family == "market_mayhem" and abs((rd - d).days) <= 3), None)
        if ref:
            sections.append(Section(hook="narrative-ref", title="Narrative", body=ph["narrative_ref"].format(ref=ref)))
            sources.append(Source(label="Market Mayhem narrative (unverified)", ref=ref))
            inputs.append(ref)
        title = fam["title"].format(lead=lead_label, regime=reg, month=d.strftime("%B %Y"),
                                    quarter=f"Q{(d.month - 1) // 3 + 1} {d.year}")
        return Edition(
            family=family, date=d, title=title, summary=f"{lead_body} {ph['regime_section'].format(vix=vix, regime=reg)}",
            sections=sections, sources=sources, estimated=True,
            provenance_trace=[ProvenanceRecord(generator=GENERATOR, inputs=inputs, timestamp=f"{d.isoformat()}T21:00:00Z",
                                               version=VERSION)],
            **kw,
        )
