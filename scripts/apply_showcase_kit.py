"""Point showcase terminals at the shared kit (showcase/kit/adam-kit.{css,js}).

For every manifest entry with role terminal or duplicate, inserts (or refreshes) one marked block before
``</head>`` and sets ``data-adam-role``/``data-adam-family``/``data-adam-canonical`` on ``<html>``, so
agents get the operator JSON view (``?view=json``, Alt+J) and ``data-*`` hooks. Idempotent; reports are
left untouched. Run after ``scripts/build_dashboard_manifest.py``.

Usage:
    python scripts/apply_showcase_kit.py [--root showcase] [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from html import escape
from pathlib import Path

_BLOCK_RE = re.compile(r"<!-- adam-kit -->.*?<!-- /adam-kit -->\n?", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<html\b[^>]*>", re.IGNORECASE)
_DATA_ATTR_RE = re.compile(r'\sdata-adam-(?:role|family|canonical)="[^"]*"')
_ROLES = {"terminal", "duplicate"}


def _kit_dir(root: Path) -> Path:
    """``kit/`` sits in ``showcase/``; a sub-catalog root (e.g. ``showcase/dashboards``) uses its parent's."""
    return next((d / "kit" for d in (root, *root.parents) if (d / "kit" / "adam-kit.js").exists()), root / "kit")


def _kit_block(page: Path, root: Path) -> str:
    rel = Path(os.path.relpath(_kit_dir(root), page.parent)).as_posix()
    return (f'<!-- adam-kit -->\n<link rel="stylesheet" href="{rel}/adam-kit.css">\n'
            f'<script src="{rel}/adam-kit.js"></script>\n<!-- /adam-kit -->')


def _html_tag(tag: str, entry: dict, page: Path, root: Path) -> str:
    attrs = {"role": entry["role"], "family": entry["family"]}
    if entry.get("canonical"):
        attrs["canonical"] = Path(os.path.relpath(root / entry["canonical"], page.parent)).as_posix()
    tag = _DATA_ATTR_RE.sub("", tag)
    extra = "".join(f' data-adam-{k}="{escape(v)}"' for k, v in attrs.items())
    return tag[:-1].rstrip() + extra + ">"


def apply(page: Path, entry: dict, root: Path) -> str | None:
    """Return the updated page text, or None when the page has no <html>/</head> to anchor on."""
    text = page.read_text(encoding="utf-8")
    head_end = text.lower().find("</head>")
    tag = _HTML_TAG_RE.search(text)
    if head_end < 0 or not tag:
        return None
    text = _BLOCK_RE.sub("", text)
    head_end = text.lower().find("</head>")
    text = text[:head_end] + _kit_block(page, root) + "\n" + text[head_end:]
    tag = _HTML_TAG_RE.search(text)
    return text[:tag.start()] + _html_tag(tag.group(0), entry, page, root) + text[tag.end():]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="showcase", type=Path)
    parser.add_argument("--check", action="store_true", help="exit 1 if any page would change")
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    changed = skipped = 0
    for entry in manifest["dashboards"]:
        if entry.get("role") not in _ROLES:
            continue
        page = args.root / entry["path"]
        new = apply(page, entry, args.root)
        if new is None:
            skipped += 1
        elif new != page.read_text(encoding="utf-8"):
            changed += 1
            if not args.check:
                page.write_text(new, encoding="utf-8")
    print(f"{'would update' if args.check else 'updated'} {changed} pages; {skipped} without <html>/</head>")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
