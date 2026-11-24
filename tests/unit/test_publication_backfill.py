import importlib.util
import json
from datetime import date
from pathlib import Path

import pytest
import yaml

from core.publication.backfill import EditionBuilder, ScenarioPath, load_config
from core.schemas.publication import ESTIMATED_LABEL, Edition

ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / "showcase" / "data" / "publications"
CHECKED_FAMILIES = ("market_mayhem", "daily_briefing", "market_pulse", "house_view")


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "publications" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scenario_path_is_reproducible_and_pinned():
    bsp = _load_script("build_scenario_path")
    cfg = yaml.safe_load((ROOT / "config/scenarios/2026H2_path.yaml").read_text())
    rebuilt = bsp.build(cfg, "config/scenarios/2026H2_path.yaml")
    committed = json.loads((ROOT / "data/scenarios/2026H2_path.json").read_text())
    assert rebuilt == committed and committed["_estimated"] is True
    path = ScenarioPath(committed)
    for name, spec in cfg["series"].items():
        for d, level in (spec["anchors"] or {}).items():
            assert path.value(name, date.fromisoformat(d)) == pytest.approx(level, abs=10 ** -spec["decimals"])


def test_every_edition_validates_and_is_labelled():
    files = sorted(PUB.glob("*/*.json"))
    assert len(files) > 200
    for f in files:
        edition = Edition.model_validate_json(f.read_text())
        assert edition.estimated and edition.provenance_trace
        assert ESTIMATED_LABEL in f.with_suffix(".md").read_text()


def test_index_points_at_existing_files():
    index = json.loads((PUB / "index.json").read_text())
    for entry in index["editions"]:
        assert (ROOT / entry["json"]).exists() and (ROOT / entry["md"]).exists()
        assert entry["html"] and ESTIMATED_LABEL in (ROOT / entry["html"]).read_text()


def test_backfill_is_deterministic():
    from core.publication.review import review_edition

    builder, prior = EditionBuilder(ScenarioPath.load(), load_config()), None
    for f in sorted((PUB / "house_view").glob("*.json")):
        stored = json.loads(f.read_text())
        draft = builder.build("house_view", date.fromisoformat(stored["date"]))
        review = stored.pop("review")
        assert draft.to_record() | {"review": {}} == stored | {"review": {}}
        assert review_edition(draft, prior) == review
        prior = draft


def test_manifest_has_no_gap_beyond_twice_cadence_and_reaches_november():
    cfg = load_config()
    lo, hi = cfg["window"]["start"], cfg["window"]["end"]
    pages = json.loads((ROOT / "showcase/manifest.json").read_text())["dashboards"]
    for family in CHECKED_FAMILIES:
        dates = sorted({date.fromisoformat(p["date"]) for p in pages
                        if p["family"] == family and len(p.get("date") or "") == 10 and lo <= p["date"] <= hi})
        gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
        assert max(gaps) <= 2 * cfg["families"][family]["cadence_days"], family
        assert dates[-1].strftime("%Y-%m") == "2026-11", family


def test_price_targets_latest_matches_engine():
    from core.publication.valuation import compute_targets

    latest = json.loads((PUB / "price_targets_latest.json").read_text())
    assert latest["_estimated"] is True and latest["label"] == ESTIMATED_LABEL
    expected = {t.ticker: t for t in compute_targets(latest["risk_free_rate_pct"])}
    assert {t["ticker"] for t in latest["targets"]} == set(expected)
    for t in latest["targets"]:
        e = expected[t["ticker"]]
        assert (t["bear"], t["base"], t["bull"], t["weighted"]) == (e.bear, e.base, e.bull, e.weighted)
        assert (ROOT / t["inputs_ref"]).exists()


def test_coverage_reuses_credit_memo_tickers():
    cov = yaml.safe_load((ROOT / "config/valuation/coverage_2026.yaml").read_text())
    for ticker, spec in cov["coverage"].items():
        assert (ROOT / spec["memo"]).exists() and ticker in spec["memo"]
