"""Build the showcase dashboard catalog for human and machine operators.

Scans ``<root>/*.html`` and writes:
  - ``<root>/manifest.json``: machine-readable catalog (path, title, kind, family, edition date).
  - ``<root>/<hub>``: searchable hub with the catalog embedded, so it works over file://.

``kind`` is the page shape (dashboard, terminal, viewer, index); ``family`` is the template family
(Market Mayhem, Daily Briefing, Market Pulse, House View, Adam OS, ...) so a page can be copied as a
starting template for its family. Idempotent: rerun after adding, moving or renaming a page.

Usage:
    python scripts/build_dashboard_manifest.py                                 # showcase/dashboards
    python scripts/build_dashboard_manifest.py --root showcase --hub catalog.html \\
        --include dashboards apps portal status_terminal events widgets templates archive_v2  # unified showcase
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_KINDS = (
    ("index", ("index", "hub", "readme")),
    ("terminal", ("terminal", "_os", "ide", "studio", "workbench")),
    ("viewer", ("viewer", "visualizer", "map", "graph", "gallery", "explorer", "browser")),
)
# Ordered: first match wins, so specific families precede broad ones.
_FAMILIES = tuple((name, re.compile(rx)) for name, rx in (
    ("market_mayhem", r"market_mayhem|mkt_myhm|^mm(\d|_|$)"),
    ("daily_briefing", r"daily_briefing|adam_daily|^brief$"),
    ("market_pulse", r"market_pulse|macro_pulse"),
    ("house_view", r"house_view"),
    ("newsletter", r"newsletter|weekly_recap"),
    ("afos", r"afos"),
    ("adam_os", r"adam_v\d|(^|_)os($|_)|terminal"),
    ("credit", r"credit|snc|covenant|lending|banking|distress"),
    ("scenario", r"counterfactual|crisis|scenario|wargame|war_room|simulat|monte_carlo|stress"),
    ("research", r"report|review|outlook|thematic|tech_watch|conviction|macro_update|geopolitic|equity_research|deep_dive|sector"),
    ("eval_governance", r"eval|governance|policy|adversarial|harness|audit|provenance"),
    ("agents", r"agent|swarm|neural|brain|cortex|nexus"),
    ("operations", r"mission_control|command_center|sentinel|monitor|radar|risk"),
    ("hub", r"hub|index|landing|navigator|archive|library|catalog"),
))
_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
_YMD_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_]?(\d{2})[-_]?(\d{2})(?!\d)")
_MDY_RE = re.compile(r"(?<!\d)(\d{2})(\d{2})((?:19|20)\d{2})(?!\d)")
_MON_Y_RE = re.compile(r"(?:^|_)(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*_(?:(\d{1,2})_)?((?:19|20)\d{2})")
_HUB = "index.html"
_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "showcase_catalog.html.tmpl"


def classify(name: str) -> str:
    lowered = name.lower()
    for kind, needles in _KINDS:
        if any(n in lowered for n in needles):
            return kind
    return "dashboard"


def family(name: str) -> str:
    lowered = name.lower()
    return next((fam for fam, rx in _FAMILIES if rx.search(lowered)), "other")


def edition_date(name: str) -> str | None:
    """ISO date encoded in a file name (2025_01_14, 20251210, 03142026, dec_2024), if any."""
    lowered = name.lower()
    for rx, order in ((_YMD_RE, "ymd"), (_MDY_RE, "mdy")):
        m = rx.search(lowered)
        if m:
            y, mo, d = (m.group(1), m.group(2), m.group(3)) if order == "ymd" else (m.group(3), m.group(1), m.group(2))
            if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
                return f"{y}-{mo}-{d}"
    m = _MON_Y_RE.search(lowered)
    if m:
        day = f"-{int(m.group(2)):02d}" if m.group(2) else ""
        return f"{m.group(3)}-{_MONTHS[m.group(1)]:02d}{day}"
    return None


def _entry(page: Path, rel: str, section: str) -> dict:
    match = _TITLE_RE.search(page.read_text(encoding="utf-8", errors="ignore"))
    title = html.unescape(" ".join(match.group(1).split())) if match else page.stem
    entry = {"path": rel, "title": title, "kind": classify(page.stem), "family": family(page.stem), "section": section}
    date = edition_date(page.stem)
    if date:
        entry["date"] = date
    return entry


def build_manifest(root: Path, hub: str = _HUB, include: tuple[str, ...] = ()) -> dict:
    """Catalog ``root/*.html`` plus ``root/<sub>/*.html`` for each ``include`` subdirectory."""
    entries = [_entry(p, p.name, "root") for p in sorted(root.glob("*.html")) if p.name != hub]
    for sub in include:
        entries += [_entry(p, f"{sub}/{p.name}", sub) for p in sorted((root / sub).glob("*.html"))]
    return {"generator": "scripts/build_dashboard_manifest.py", "count": len(entries), "dashboards": entries}


def render_hub(manifest: dict, title: str, root: str = "showcase") -> str:
    """``root`` is the repo-relative catalog directory; copied paths are ``root/<entry path>``."""
    embedded = json.dumps(manifest, separators=(",", ":")).replace("</", "<\\/")
    page = _TEMPLATE_PATH.read_text(encoding="utf-8").replace("__TITLE__", title).replace("__ROOT__", html.escape(root))
    return page.replace("__MANIFEST__", embedded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="showcase/dashboards", type=Path)
    parser.add_argument("--hub", default=_HUB, help="hub page file name written into --root")
    parser.add_argument("--include", nargs="*", default=[], help="subdirectories of --root to catalog too")
    args = parser.parse_args()
    root, hub = args.root, args.hub
    manifest = build_manifest(root, hub, tuple(args.include))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    title = "Dashboard Catalog" if root.name == "dashboards" else "Unified Showcase"
    (root / hub).write_text(render_hub(manifest, title, root.as_posix().rstrip("/")), encoding="utf-8")
    print(f"Wrote {manifest['count']} pages to {root}/manifest.json and {root}/{hub}")


if __name__ == "__main__":
    main()
