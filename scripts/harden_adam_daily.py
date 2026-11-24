"""Escape dynamic HTML in the Adam Daily pages (showcase/data/adam_daily/**/*.html).

Idempotent rewrite, re-run after adding a page:
* inserts the shared kit (``<!-- adam-kit -->`` block) so pages can call ``AdamKit.esc``;
* wraps simple ``${name}`` / ``${obj.field}`` interpolations inside ``.innerHTML = `...` `` /
  ``+= `...` `` template literals with ``AdamKit.esc(...)`` (names that carry pre-built markup, e.g.
  ``rowsHtml``, are left alone);
* escapes Markdown before ``marked.parse(...)`` and writes raw LLM ``response`` strings as text.

    python scripts/harden_adam_daily.py [--check]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "showcase" / "data" / "adam_daily"
KIT = ROOT / "showcase" / "kit"
_SINK_RE = re.compile(r"\.innerHTML\s*\+?=\s*`")
_SIMPLE_RE = re.compile(r"^\s*[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*|\[\d+\])*\s*$")
_MARKUP_RE = re.compile(r"html|svg|icon|badge|rows?$|cells|markup|tpl|template|spark|bars?$", re.IGNORECASE)
_MARKED_RE = re.compile(r"marked\.parse\((?!AdamKit\.esc\()([^()]*(?:\([^()]*\))?[^()]*)\)")
_RESPONSE_RE = re.compile(r"\.innerHTML\s*=\s*response\s*;")


def _literal_end(text: str, start: int) -> tuple[int, list[tuple[int, int]]]:
    """Index of the closing backtick of the literal opened at ``start`` and its top-level ``${}`` spans."""
    i, spans = start + 1, []
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "`":
            return i, spans
        if text.startswith("${", i):
            depth, j = 1, i + 2
            while j < len(text) and depth:
                if text[j] == "`":
                    j = _literal_end(text, j)[0]
                elif text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            spans.append((i + 2, j - 1))
            i = j
            continue
        i += 1
    return len(text), spans


def unescaped_interpolations(text: str) -> list[str]:
    """Simple interpolations inside innerHTML literals that are not wrapped in AdamKit.esc."""
    out = []
    for m in _SINK_RE.finditer(text):
        _, spans = _literal_end(text, m.end() - 1)
        out += [text[a:b].strip() for a, b in spans if _SIMPLE_RE.match(text[a:b]) and not _MARKUP_RE.search(text[a:b])]
    return out


def harden(text: str, kit_href: str) -> str:
    pieces, last = [], 0
    for m in _SINK_RE.finditer(text):
        if m.start() < last:
            continue
        end, spans = _literal_end(text, m.end() - 1)
        for a, b in spans:
            expr = text[a:b]
            if _SIMPLE_RE.match(expr) and not _MARKUP_RE.search(expr):
                pieces += [text[last:a], f"AdamKit.esc({expr.strip()})"]
                last = b
        pieces.append(text[last:end + 1])
        last = end + 1
    text = "".join(pieces) + text[last:]
    text = _MARKED_RE.sub(lambda m: f"marked.parse(AdamKit.esc({m.group(1)}))", text)
    text = _RESPONSE_RE.sub(".textContent = response;", text)
    if "<!-- adam-kit -->" not in text and "</head>" in text:
        block = (f'<!-- adam-kit -->\n<link rel="stylesheet" href="{kit_href}/adam-kit.css">\n'
                 f'<script src="{kit_href}/adam-kit.js"></script>\n<!-- /adam-kit -->\n')
        text = text.replace("</head>", block + "</head>", 1)
    return text


def pages() -> list[Path]:
    return sorted(DAILY.rglob("*.html"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if any page would change")
    args = ap.parse_args()
    changed = []
    for page in pages():
        text = page.read_text(encoding="utf-8")
        new = harden(text, Path(os.path.relpath(KIT, page.parent)).as_posix())
        if new != text:
            changed.append(page.relative_to(ROOT).as_posix())
            if not args.check:
                page.write_text(new, encoding="utf-8")
    print(f"{'would update' if args.check else 'updated'} {len(changed)} of {len(pages())} pages")
    return int(args.check and bool(changed))


if __name__ == "__main__":
    sys.exit(main())
