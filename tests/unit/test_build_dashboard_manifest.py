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
        "section": "root",
        "role": "report",
        "date": "2025-01-14",
    }
    json.dumps(manifest)


def test_include_subdirectories_and_render_hub(tmp_path):
    (tmp_path / "Market_Pulse_2025_03_01.html").write_text("<title>Pulse</title>")
    (tmp_path / "dashboards").mkdir()
    (tmp_path / "dashboards" / "adam_v30_terminal.html").write_text("<title>Terminal</title>")
    manifest = bdm.build_manifest(tmp_path, "catalog.html", ("dashboards",))
    by_path = {e["path"]: e for e in manifest["dashboards"]}
    assert by_path["dashboards/adam_v30_terminal.html"]["section"] == "dashboards"
    assert by_path["dashboards/adam_v30_terminal.html"]["family"] == "adam_os"
    page = bdm.render_hub(manifest, "Unified Showcase")
    assert "__MANIFEST__" not in page and "Unified Showcase" in page
    assert 'data-root="showcase"' in page
    assert 'data-root="showcase/dashboards"' in bdm.render_hub(manifest, "Dashboard Catalog", "showcase/dashboards")
    assert "dashboards/adam_v30_terminal.html" in page


_KIT_SPEC = importlib.util.spec_from_file_location(
    "apply_showcase_kit", Path(__file__).resolve().parents[2] / "scripts" / "apply_showcase_kit.py"
)
kit = importlib.util.module_from_spec(_KIT_SPEC)
_KIT_SPEC.loader.exec_module(kit)


def test_roles_canonical_groups_and_exact_duplicates(tmp_path):
    live = "<title>T</title><script>1</script>"
    (tmp_path / "a_terminal.html").write_text(live)
    (tmp_path / "b_terminal.html").write_text(live)
    (tmp_path / "old_os.html").write_text(live + "<!-- v1 -->")
    (tmp_path / "Daily_Briefing_2025_01_14.html").write_text(live + "<!-- brief -->")
    (tmp_path / "static.html").write_text("<title>S</title>")
    (tmp_path / "canonical.json").write_text(json.dumps({"os": {"canonical": "a_terminal.html", "members": ["old_os.html"]}}))
    by_path = {e["path"]: e for e in bdm.build_manifest(tmp_path)["dashboards"]}
    assert by_path["a_terminal.html"]["role"] == "terminal"
    assert (by_path["b_terminal.html"]["role"], by_path["b_terminal.html"]["canonical"]) == ("duplicate", "a_terminal.html")
    assert (by_path["old_os.html"]["role"], by_path["old_os.html"]["canonical"]) == ("duplicate", "a_terminal.html")
    assert by_path["Daily_Briefing_2025_01_14.html"]["role"] == "report"
    assert by_path["static.html"]["role"] == "report"
    assert all("_sha" not in e for e in by_path.values())


def test_apply_kit_is_idempotent_and_relative(tmp_path):
    (tmp_path / "dashboards").mkdir()
    page = tmp_path / "dashboards" / "old.html"
    page.write_text('<html lang="en"><head><title>x</title></head><body></body></html>')
    entry = {"path": "dashboards/old.html", "role": "duplicate", "family": "adam_os", "canonical": "catalog.html"}
    once = kit.apply(page, entry, tmp_path)
    page.write_text(once)
    assert kit.apply(page, entry, tmp_path) == once
    assert '<script src="../kit/adam-kit.js"></script>' in once
    assert '<html lang="en" data-adam-role="duplicate" data-adam-family="adam_os" data-adam-canonical="../catalog.html">' in once
    assert once.count("<!-- adam-kit -->") == 1


def test_duplicates_survive_kit_and_parent_canonical(tmp_path):
    sub = tmp_path / "dashboards"
    sub.mkdir()
    page = '<html lang="en"><head><title>T</title><script>1</script></head></html>'
    (sub / "a_terminal.html").write_text(page)
    (sub / "b_terminal.html").write_text(page)
    (sub / "old_os.html").write_text(page.replace("1", "2"))
    (tmp_path / "canonical.json").write_text(
        json.dumps({"os": {"canonical": "dashboards/a_terminal.html", "members": ["dashboards/old_os.html"]}})
    )
    for _ in range(2):
        entries = {e["path"]: e for e in bdm.build_manifest(sub)["dashboards"]}
        for e in entries.values():
            (sub / e["path"]).write_text(kit.apply(sub / e["path"], e, sub))
    assert entries["b_terminal.html"]["canonical"] == "a_terminal.html"
    assert entries["old_os.html"]["canonical"] == "a_terminal.html"
