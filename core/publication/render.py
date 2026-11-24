"""Render a validated ``Edition`` to Markdown and HTML with the per-family templates.

Templates live in ``showcase/templates/publication/<family>.{md,html}.j2`` (presentation stays in
``showcase/``). HTML is rendered with Jinja autoescape; the edition record is embedded verbatim as
``<script id="adam-operator-data" type="application/json">`` so ``AdamKit.snapshot()`` / ``?view=json``
return exactly the data the page renders.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from core.schemas.publication import ESTIMATED_LABEL, Edition

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO_ROOT / "showcase" / "templates" / "publication"
KIT_DIR = REPO_ROOT / "showcase" / "kit"

FAMILY_LABELS = {
    "market_mayhem": "Market Mayhem", "daily_briefing": "Daily Briefing", "market_pulse": "Market Pulse",
    "house_view": "House View", "equity_research": "Equity Research", "sector_outlook": "Sector Outlook",
    "credit_memo": "Credit Memo", "deep_dive": "Deep Dive",
}


def _md_escape(value: object) -> str:
    """Neutralise raw HTML tags in Markdown output (Markdown renderers pass inline HTML through)."""
    return str(value if value is not None else "").replace("<", "&lt;").replace(">", "&gt;")


def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def _env() -> Environment:
    """HTML templates (``*.html.j2``) autoescape; Markdown templates escape via the ``md`` filter."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)),
                      autoescape=select_autoescape(enabled_extensions=("html.j2",), default=False),
                      undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    env.filters["md"] = _md_escape
    env.filters["num"] = _fmt
    return env


_ENV = _env()


def _context(edition: Edition, **extra) -> dict:
    return {
        "e": edition,
        "record": edition.to_record(),
        "family_label": FAMILY_LABELS[edition.family],
        "estimated_label": ESTIMATED_LABEL,
        **extra,
    }


def render_markdown(edition: Edition) -> str:
    return _ENV.get_template(f"{edition.family}.md.j2").render(**_context(edition))


def render_html(edition: Edition, kit_href: str = "kit") -> str:
    """``kit_href`` is the relative URL from the output page to ``showcase/kit``."""
    return _ENV.get_template(f"{edition.family}.html.j2").render(**_context(edition, kit_href=kit_href))


def kit_href_for(html_path: Path) -> str:
    return Path(os.path.relpath(KIT_DIR, html_path.resolve().parent)).as_posix()


def write_edition(edition: Edition, json_path: Path, md_path: Optional[Path] = None,
                  html_path: Optional[Path] = None) -> None:
    """Write JSON (source of truth), then the Markdown and HTML renderings derived from it."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(edition.to_record(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(edition), encoding="utf-8")
    if html_path:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(render_html(edition, kit_href_for(html_path)), encoding="utf-8")
