import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.publication.render import TEMPLATE_DIR, render_html, render_markdown, write_edition
from core.schemas.publication import ESTIMATED_LABEL, FAMILIES, Edition, PriceTarget

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "publication_edition.json"


@pytest.fixture
def record():
    return json.loads(FIXTURE.read_text())


def _edition(record, family):
    return Edition.model_validate({**record, "family": family})


def test_round_trip_preserves_record(record):
    edition = Edition.model_validate(record)
    dumped = edition.to_record()
    assert dumped["title"] == record["title"] and dumped["_estimated"] is True
    assert dumped["price_targets"][0]["_estimated"] is True and "estimated" not in dumped["price_targets"][0]
    assert Edition.model_validate(dumped).to_record() == dumped
    assert Edition.model_validate_json(json.dumps(edition.to_record())) == edition


def test_estimated_propagates_from_parts(record):
    record["_estimated"] = False
    assert Edition.model_validate(record).estimated is True


def test_provenance_required(record):
    record["provenance_trace"] = []
    with pytest.raises(ValidationError):
        Edition.model_validate(record)


def test_duplicate_section_hooks_rejected(record):
    record["sections"][1]["hook"] = "lead"
    with pytest.raises(ValidationError):
        Edition.model_validate(record)


@pytest.mark.parametrize("bad", [{"bear": 600.0}, {"weights": {"base": 0.5, "bull": 0.5, "bear": 0.5}}])
def test_price_target_invariants(record, bad):
    with pytest.raises(ValidationError):
        PriceTarget.model_validate({**record["price_targets"][0], **bad})


def test_weighted_target(record):
    assert PriceTarget.model_validate(record["price_targets"][0]).weighted == 445.0


def test_weighted_target_rounds_ties_half_up(record):
    tie = {**record["price_targets"][0], "bear": 67.32, "base": 103.22, "bull": 134.22}
    assert PriceTarget.model_validate(tie).weighted == 102.0


def test_every_family_has_templates():
    for family in FAMILIES:
        assert (TEMPLATE_DIR / f"{family}.html.j2").exists()
        assert (TEMPLATE_DIR / f"{family}.md.j2").exists()


@pytest.mark.parametrize("family", FAMILIES)
def test_html_template_renders_fixture(record, family):
    html = render_html(_edition(record, family))
    assert f'data-adam-family="{family}"' in html
    assert ESTIMATED_LABEL in html
    for hook in ("masthead", "lead", "outlook", "market-snapshot", "calls", "price-targets", "scenarios", "risks", "provenance"):
        assert f'data-adam-hook="{hook}"' in html
    for section in re.findall(r"<section\b[^>]*>", html):
        assert "data-adam-hook=" in section
    embedded = re.search(r'<script id="adam-operator-data" type="application/json">(.*?)</script>', html, re.S)
    assert json.loads(embedded.group(1)) == _edition(record, family).to_record()


@pytest.mark.parametrize("family", FAMILIES)
def test_markdown_template_renders_fixture(record, family):
    md = render_markdown(_edition(record, family))
    assert md.startswith("# ") and ESTIMATED_LABEL in md
    assert "| MSFT" in md and "## Outlook" in md
    assert "| 12m |" in md and "spx 7,600.00" in md


def test_markdown_list_items_stay_on_separate_lines(record):
    record["sources"].append({"label": "Second", "ref": "docs/PROVENANCE.md", "as_of": "2026-11-12"})
    record["risks"].append({"name": "Second risk", "probability": 0.1, "impact": "low", "description": "x"})
    md = render_markdown(Edition.model_validate(record))
    lines = md.splitlines()
    assert "- **Second risk** — p=10%, impact low: x" in lines
    assert "- Second — `docs/PROVENANCE.md` (as of 2026-11-12)" in lines
    assert any(line.startswith("- generator `tests/fixtures`") for line in lines)


def test_estimated_risk_flags_edition(record):
    for part in ("market_snapshot", "calls", "price_targets", "scenarios"):
        for item in record[part]:
            item["_estimated"] = False
    record["_estimated"] = False
    record["risks"][0]["_estimated"] = True
    edition = Edition.model_validate(record)
    assert edition.estimated and "Term-premium shock</strong>*" in render_html(edition)


@pytest.mark.parametrize("family", FAMILIES)
def test_rendered_output_escapes_injected_script(record, family):
    edition = _edition(record, family)
    html = render_html(edition)
    scripts = re.findall(r"<script\b[^>]*>", html, re.IGNORECASE)
    allowed = ('src="kit/adam-kit.js"', 'src="kit/adam-publication.js"', 'id="adam-operator-data"')
    assert all(any(a in s for a in allowed) for s in scripts)
    assert "<script>alert" not in html.lower() and "<img src=x" not in html.lower()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>" not in render_markdown(edition)


def test_write_edition_emits_json_md_html(tmp_path, record):
    edition = Edition.model_validate(record)
    write_edition(edition, tmp_path / "e.json", tmp_path / "e.md", tmp_path / "out" / "e.html")
    assert Edition.model_validate_json((tmp_path / "e.json").read_text()) == edition
    assert (tmp_path / "e.md").read_text().startswith("# ")
    assert "adam-kit.js" in (tmp_path / "out" / "e.html").read_text()
