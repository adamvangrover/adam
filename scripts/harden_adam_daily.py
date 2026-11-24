"""Escape dynamic HTML in the Adam Daily pages (showcase/data/adam_daily/**/*.html).

Idempotent rewrite, re-run after adding a page:
* inserts the shared kit (``<!-- adam-kit -->`` block) so pages can call ``AdamKit.esc``;
* wraps simple ``${name}`` / ``${obj.field}`` interpolations inside ``.innerHTML = `...` `` /
  ``+= `...` `` template literals with ``AdamKit.esc(...)`` (names that carry pre-built markup, e.g.
  ``rowsHtml``, or variables assigned a ``<...>`` string literal, are left alone);
* renders Markdown through ``AdamKit.markdown(...)`` (raw HTML escaped, fences kept) and writes raw LLM
  ``response`` strings as text.

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
_MARKED_RE = re.compile(r"marked\.parse\(AdamKit\.esc\(([^()]*(?:\([^()]*\))?[^()]*)\)\)|marked\.parse\(")
_MARKUP_ASSIGN_RE = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\+?=\s*(?:[^;`'\"]*\?\s*)?[`'\"]\s*<")
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


def _markup_vars(text: str) -> set[str]:
    """Variables the page assigns an HTML string literal to (pre-built markup, not data)."""
    return set(_MARKUP_ASSIGN_RE.findall(text))


def _is_data(expr: str, markup: set[str]) -> bool:
    return bool(_SIMPLE_RE.match(expr)) and not _MARKUP_RE.search(expr) and expr.strip() not in markup


def unescaped_interpolations(text: str) -> list[str]:
    """Simple interpolations inside innerHTML literals that are not wrapped in AdamKit.esc."""
    out, markup = [], _markup_vars(text)
    for m in _SINK_RE.finditer(text):
        _, spans = _literal_end(text, m.end() - 1)
        out += [text[a:b].strip() for a, b in spans if _is_data(text[a:b], markup)]
    return out


def harden(text: str, kit_href: str) -> str:
    markup = _markup_vars(text)
    for name in markup:
        text = text.replace("${AdamKit.esc(" + name + ")}", "${" + name + "}")
    pieces, last = [], 0
    for m in _SINK_RE.finditer(text):
        if m.start() < last:
            continue
        end, spans = _literal_end(text, m.end() - 1)
        for a, b in spans:
            expr = text[a:b]
            if _is_data(expr, markup):
                pieces += [text[last:a], f"AdamKit.esc({expr.strip()})"]
                last = b
        pieces.append(text[last:end + 1])
        last = end + 1
    text = "".join(pieces) + text[last:]
    text = _MARKED_RE.sub(lambda m: f"AdamKit.markdown({m.group(1)})" if m.group(1) is not None else "AdamKit.markdown(", text)
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
