import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_dashboard_manifest", Path(__file__).resolve().parents[2] / "scripts" / "build_dashboard_manifest.py"
)
bdm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bdm)


@pytest.mark.parametrize(
    "stem, family, date",
    [
        ("Daily_Briefing_2025_01_14", "daily_briefing", "2025-01-14"),
        ("Market_Pulse_2026_03_20", "market_pulse", "2026-03-20"),
        ("house_view_20251210", "house_view", "2025-12-10"),
        ("MM11132025", "market_mayhem", "2025-11-13"),
        ("1987-10-23_Market_Mayhem", "market_mayhem", "1987-10-23"),
        ("newsletter_market_mayhem_dec_2024", "market_mayhem", "2024-12"),
        ("newsletter_market_mayhem_cro_ib_week_ahead_mar_16_2026", "market_mayhem", "2026-03-16"),
        ("adam_v31_0_cyber_retro_institutional_os", "adam_os", None),
        ("analyst_os", "adam_os", None),
        ("comprehensive_credit_dashboard", "credit", None),
        ("summary", "other", None),
    ],
)
def test_family_and_edition_date(stem, family, date):
    assert bdm.family(stem) == family
    assert bdm.edition_date(stem) == date


def test_invalid_month_is_not_a_date():
    assert bdm.edition_date("report_20251399") is None


def test_build_manifest_skips_hub_and_records_fields(tmp_path):
    (tmp_path / "Daily_Briefing_2025_01_14.html").write_text("<title>Brief</title>")
    (tmp_path / "catalog.html").write_text("<title>hub</title>")
    manifest = bdm.build_manifest(tmp_path, "catalog.html")
    assert manifest["count"] == 1
    assert manifest["dashboards"][0] == {
        "path": "Daily_Briefing_2025_01_14.html",
        "title": "Brief",
        "kind": "dashboard",
        "family": "daily_briefing",
        "date": "2025-01-14",
    }
    json.dumps(manifest)
