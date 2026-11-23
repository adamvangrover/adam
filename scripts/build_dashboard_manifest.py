"""Build the showcase dashboard catalog for human and machine operators.

Scans ``<root>/*.html`` and writes:
  - ``<root>/manifest.json``: machine-readable catalog (path, title, kind, family, edition date).
  - ``<root>/<hub>``: searchable hub with the catalog embedded, so it works over file://.

``role`` is terminal (interactive), report (static edition/research) or duplicate (an exact copy, or a
member of a ``<root>/canonical.json`` group; ``canonical`` names the page to use instead).
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
import hashlib
import html
import json
import os
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
# Kit markup written by scripts/apply_showcase_kit.py; stripped before hashing so it cannot mask duplicates.
_KIT_RE = re.compile(r"<!-- adam-kit -->.*?<!-- /adam-kit -->\n?|\sdata-adam-(?:role|family|canonical)=\"[^\"]*\"", re.DOTALL)
_REPORT_FAMILIES = {"market_mayhem", "daily_briefing", "market_pulse", "house_view", "newsletter", "research"}
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


def role(fam: str, text: str) -> str:
    return "report" if fam in _REPORT_FAMILIES or "<script" not in text.lower() else "terminal"


def _canonical_map(root: Path) -> dict[str, str]:
    """Member -> canonical, both relative to ``root``; ``canonical.json`` may sit in ``root`` or its parent."""
    path = next((d / "canonical.json" for d in (root, root.parent) if (d / "canonical.json").exists()), None)
    if path is None:
        return {}
    base, out = path.parent, {}
    for key, group in json.loads(path.read_text(encoding="utf-8")).items():
        if key.startswith("_"):
            continue
        target = Path(os.path.relpath(base / group["canonical"], root)).as_posix()
        for member in group["members"]:
            rel = Path(os.path.relpath(base / member, root)).as_posix()
            if not rel.startswith("../"):
                out[rel] = target
    return out


def _assign_roles(entries: list[dict], root: Path) -> None:
    canonical, seen = _canonical_map(root), {}
    for e in entries:
        digest = e.pop("_sha")
        target = canonical.get(e["path"]) or seen.get(digest)
        if target:
            e["role"], e["canonical"] = "duplicate", target
        seen.setdefault(digest, e["path"])


def _entry(page: Path, rel: str, section: str) -> dict:
    text = page.read_text(encoding="utf-8", errors="ignore")
    match = _TITLE_RE.search(text)
    title = html.unescape(" ".join(match.group(1).split())) if match else page.stem
    entry = {"path": rel, "title": title, "kind": classify(page.stem), "family": family(page.stem), "section": section}
    entry["role"] = role(entry["family"], text)
    entry["_sha"] = hashlib.sha256(_KIT_RE.sub("", text).encode("utf-8")).hexdigest()
    date = edition_date(page.stem)
    if date:
        entry["date"] = date
    return entry


def build_manifest(root: Path, hub: str = _HUB, include: tuple[str, ...] = ()) -> dict:
    """Catalog ``root/*.html`` plus ``root/<sub>/*.html`` for each ``include`` subdirectory."""
    entries = [_entry(p, p.name, "root") for p in sorted(root.glob("*.html")) if p.name != hub]
    for sub in include:
        entries += [_entry(p, f"{sub}/{p.name}", sub) for p in sorted((root / sub).glob("*.html"))]
    _assign_roles(entries, root)
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
