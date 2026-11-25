"""Backfill Apr–Nov 2026 publication editions: data JSON -> MD -> HTML.

Reads the committed synthetic path (data/scenarios/2026H2_path.json, build it with
scripts/publications/build_scenario_path.py) and config/publications.yaml, then for every family and
edition date writes ``showcase/data/publications/<family>/<date>.{json,md}`` (source of truth) and the
HTML rendering at the family's existing path so ``showcase/manifest.json`` picks it up. Also writes
``showcase/data/publications/index.json`` and ``price_targets_latest.json``. Idempotent; never
overwrites a hand-made / earlier-generation HTML page (pages without the ``adam-edition`` meta tag).

    python scripts/publications/backfill_editions.py [--families market_mayhem daily_briefing ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.disable(logging.INFO)

from core.publication.backfill import (  # noqa: E402
    GENERATOR, PATH_FILE, VERSION, EditionBuilder, ScenarioPath, edition_dates, load_config,
)
from core.publication.render import write_edition  # noqa: E402
from core.publication.review import load_review_config, review_edition  # noqa: E402
from core.publication.valuation import COVERAGE_CONFIG, compute_targets  # noqa: E402
from core.schemas.publication import ESTIMATED_LABEL  # noqa: E402

OUT = ROOT / "showcase" / "data" / "publications"
_MM_RE = re.compile(r"market_mayhem_(\d{4})(\d{2})(\d{2})\.md$")


def mm_narratives() -> dict[date, str]:
    refs = {}
    for p in sorted((ROOT / "newsletters").glob("market_mayhem_2026*.md")):
        m = _MM_RE.search(p.name)
        if m:
            refs[date(*map(int, m.groups()))] = p.relative_to(ROOT).as_posix()
    return refs


def html_path(template: str, d: date) -> Path:
    return ROOT / template.format(Y=d.strftime("%Y"), m=d.strftime("%m"), d=d.strftime("%d"))


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--families", nargs="*", default=list(cfg["families"]))
    args = ap.parse_args()
    path = ScenarioPath.load()
    builder = EditionBuilder(path, cfg, mm_narratives())
    review_cfg = load_review_config()
    index, skipped = [], []
    for family in args.families:
        prior = None
        for d in edition_dates(family, cfg, path):
            draft = builder.build(family, d)
            edition = draft.model_copy(update={"review": review_edition(draft, prior, review_cfg)})
            prior = draft
            base = OUT / family / d.isoformat()
            html = html_path(cfg["families"][family]["html"], d)
            if html.exists() and 'name="adam-edition"' not in html.read_text(errors="ignore"):
                skipped.append(_rel(html))
                html = None
            write_edition(edition, base.with_suffix(".json"), base.with_suffix(".md"), html)
            index.append({"id": edition.edition_id, "family": family, "date": d.isoformat(), "title": edition.title,
                          "json": _rel(base.with_suffix(".json")), "md": _rel(base.with_suffix(".md")),
                          "html": _rel(html) if html else None, "_estimated": edition.estimated,
                          "tickers": [t.ticker for t in edition.price_targets],
                          "review_score": edition.review["score"], "review_verdict": edition.review["desk"]["verdict"],
                          "flags": [f["code"] for f in edition.review["flags"]]})
    end = max(path.dates)
    trace = {"generator": GENERATOR, "inputs": [PATH_FILE, COVERAGE_CONFIG], "timestamp": f"{end.isoformat()}T21:00:00Z",
             "version": VERSION}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.json").write_text(json.dumps({"_estimated": True, "label": ESTIMATED_LABEL, "editions": index,
                                                "provenance_trace": [trace]}, indent=1) + "\n")
    rfr = path.value("ust10y", end)
    latest = {"_estimated": True, "label": ESTIMATED_LABEL, "as_of": end.isoformat(), "risk_free_rate_pct": rfr,
              "targets": [t.model_dump(mode="json", by_alias=True) | {"weighted": t.weighted} for t in compute_targets(rfr)],
              "provenance_trace": [trace]}
    (OUT / "price_targets_latest.json").write_text(json.dumps(latest, indent=1) + "\n")
    print(f"wrote {len(index)} editions; kept {len(skipped)} existing pages: {skipped[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
