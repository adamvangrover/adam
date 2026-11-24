import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _mod():
    spec = importlib.util.spec_from_file_location("coverage_report", ROOT / "scripts/publications/coverage_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_coverage_report_is_fresh():
    assert (ROOT / "docs/CONTENT_COVERAGE.md").read_text() == _mod().build()


def test_coverage_lists_every_backfilled_family():
    text = _mod().build()
    for family in ("market_mayhem", "daily_briefing", "market_pulse", "house_view", "equity_research", "sector_outlook"):
        assert f"| {family} |" in text
