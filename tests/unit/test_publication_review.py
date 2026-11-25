import json
import re
from pathlib import Path

import pytest

from core.publication.render import render_html
from core.publication.review import load_review_config, review_edition, review_flags
from core.publication.review.red_team import PublicationRedTeam, red_team_review
from core.schemas.publication import Edition

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "publication_edition.json"
PUB = ROOT / "showcase" / "data" / "publications"


@pytest.fixture
def record():
    return json.loads(FIXTURE.read_text())


def test_offline_review_shape(record):
    review = review_edition(record)
    assert review["mode"] == "offline" and load_review_config()["llm"]["enabled"] is False
    assert {"flags", "desk", "red_team", "judge", "score", "panel"} <= set(review)
    assert set(review["judge"]["weights"]) == {d["name"] for d in review["judge"]["dimensions"]}
    assert review["red_team"]["flipped_scenario"]["name"] == "bear"
    assert review["red_team"]["report"].startswith("=== Red Team Exercise Report ===")


def test_desk_blocks_on_error_flags(record):
    record["sources"] = []
    review = review_edition(record)
    assert review["desk"]["verdict"] == "block"
    assert "missing_sources" in {f["code"] for f in review["flags"]}


def test_red_team_framework_accepts_custom_agent(record):
    edition = Edition.model_validate(record)
    out = red_team_review(edition, [{"code": "stale_data", "severity": "warning", "hook": "provenance", "message": "old"}],
                          load_review_config(), agent=PublicationRedTeam(load_review_config()))
    assert out["challenges"][0]["code"] == "stale_data" and out["impact_score"] > 0


def test_prior_edition_contradiction(record):
    prior = json.loads(json.dumps(record))
    prior["calls"][0].update(stance="overweight", conviction=0.9)
    record["calls"][0]["conviction"] = 0.9
    assert [f["code"] for f in review_flags(record, prior)] == ["contradicts_prior"]


def test_panel_renders_and_escapes(record):
    review = review_edition(record)
    review["panel"]["red_team"]["challenges"] = ["<script>alert(9)</script>"]
    html = render_html(Edition.model_validate({**record, "review": review}))
    for hook in ("review-panel", "review-bull", "review-bear", "review-red-team", "review-scores"):
        assert f'data-adam-hook="{hook}"' in html
    assert "Bull vs. Bear vs. Red Team" in html and "<script>alert(9)" not in html
    assert re.search(r'<script src="kit/adam-publication.js" defer>', html)


def test_every_backfilled_edition_carries_a_review():
    index = json.loads((PUB / "index.json").read_text())["editions"]
    assert index
    for entry in index:
        edition = Edition.model_validate_json((ROOT / entry["json"]).read_text())
        assert edition.review["mode"] == "offline" and 0 <= edition.review["score"] <= 1
        assert entry["review_score"] == edition.review["score"]
        assert "Bull vs. Bear vs. Red Team" in (ROOT / entry["html"]).read_text()
