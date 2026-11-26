import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHOWCASE = ROOT / "showcase"
TERMINALS = {"editions_browser": "editions", "price_target_explorer": "price_targets", "review_panel": "publication_review"}
_INTERACTIVE_RE = re.compile(r"<(select|input|button|table|tbody|section|nav)\b[^>]*>", re.IGNORECASE)


@pytest.mark.parametrize("name", TERMINALS)
def test_terminal_page_is_kit_wired_and_hooked(name):
    html = (SHOWCASE / f"{name}.html").read_text()
    assert "<!-- adam-kit -->" in html and f'src="js/{name}.js"' in html
    assert 'data-adam-family="publications"' in html and "not investment advice" in html
    for tag in _INTERACTIVE_RE.finditer(html):
        assert "data-adam-hook=" in tag.group(0), tag.group(0)
    inline = [s for s in re.findall(r"<script\b([^>]*)>(.*?)</script>", html, re.S) if "src=" not in s[0]]
    assert not inline, "terminals must not embed inline scripts / literal datasets"


@pytest.mark.parametrize("name", TERMINALS)
def test_terminal_js_reads_published_json_and_escapes(name):
    js = (SHOWCASE / "js" / f"{name}.js").read_text()
    assert "data/publications/" in js and "K.publish(" in js
    for rhs in re.findall(r"\.innerHTML\s*=\s*([^;]+);", js):
        assert rhs.lstrip().startswith(("rows.map", "K.html", "AdamKit.html")), rhs[:80]
    assert "insertAdjacentHTML" not in js or "K.html`<option" in js


def test_canonical_roles_registered():
    canonical = json.loads((SHOWCASE / "canonical.json").read_text())
    for name, role in TERMINALS.items():
        assert canonical[role]["canonical"] == f"{name}.html"
        for member in canonical[role]["members"]:
            assert (SHOWCASE / member).exists()


def test_terminal_inputs_exist():
    for path in ("data/publications/index.json", "data/publications/price_targets_latest.json"):
        assert json.loads((SHOWCASE / path).read_text())["_estimated"] is True


def test_adam_daily_pages_are_hardened():
    import importlib.util

    spec = importlib.util.spec_from_file_location("harden", ROOT / "scripts" / "harden_adam_daily.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for page in mod.pages():
        text = page.read_text(encoding="utf-8")
        assert mod.unescaped_interpolations(text) == [], page
        assert "marked.parse(" not in text, page
        for name in mod._markup_vars(text):
            assert "${AdamKit.esc(" + name + ")}" not in text, (page, name)
        assert not re.search(r"\.innerHTML\s*=\s*response\s*;", text), page
        if "</head>" in text:
            assert "<!-- adam-kit -->" in text, page


def test_harden_escapes_user_input():
    import importlib.util

    spec = importlib.util.spec_from_file_location("harden", ROOT / "scripts" / "harden_adam_daily.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = "<head></head><script>out.innerHTML += `bash: ${cmd} ${rowsHtml} ${items.map(i => `<b>${i}</b>`).join('')}`;</script>"
    out = mod.harden(src, "../kit")
    assert "${AdamKit.esc(cmd)}" in out and "${rowsHtml}" in out and mod.harden(out, "../kit") == out


def test_harden_keeps_markup_variables_and_markdown_fences():
    import importlib.util

    spec = importlib.util.spec_from_file_location("harden", ROOT / "scripts" / "harden_adam_daily.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = ("<head></head><script>let btns = ''; btns += `<button>${t}</button>`;"
           "const dot = on ? `<i></i>` : '';"
           "const md2 = AdamKit.markdown(x); const lst = xs.map(v => {\n const t = v; return `<b>${t}</b>`;\n}).join('');"
           "c.innerHTML = `${btns}${dot}${AdamKit.esc(btns)}${name}${AdamKit.esc(md2)}${lst}`; d.innerHTML = marked.parse(AdamKit.esc(md));"
           "e.innerHTML = marked.parse(raw);</script>")
    out = mod.harden(src, "../kit")
    assert "${btns}${dot}${btns}${AdamKit.esc(name)}${md2}${lst}" in out
    assert "AdamKit.markdown(md)" in out and "AdamKit.markdown(raw)" in out and "marked.parse" not in out
    assert mod.harden(out, "../kit") == out
