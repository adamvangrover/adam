import importlib.util
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHOWCASE = ROOT / "showcase"
TERMINALS = {
    "editions_browser": "editions",
    "price_target_explorer": "price_targets",
    "review_panel": "publication_review",
}
INTERACTIVE_TAGS = frozenset({"select", "input", "button", "table", "tbody", "section", "nav"})

_SAFE_HTML_PREFIX_RE = re.compile(
    r"^(?:(?:K|AdamKit)\.html`|(?:K|AdamKit)\.markdown\(|rows\.map\(|\"\"|''|``)"
)
_TAGGED_HTML_RE = re.compile(r"(?:K|AdamKit)\.html\s*$")
_SAFE_INTERP_RE = re.compile(r"^(?:K|AdamKit)\.(?:esc|markdown)\(|^(?:K|AdamKit)\.html`")
_DOM_SINK_RE = re.compile(r"\.(?:inner|outer)HTML\s*\+?=(?!=)")
_INSERT_ADJACENT_RE = re.compile(r"\.insertAdjacentHTML\s*\(")
_REGEX_PREV_KEYWORDS = frozenset(
    {"return", "case", "in", "of", "typeof", "void", "delete", "throw", "new", "else"}
)


class _TerminalHTMLInspector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.html_attrs: dict[str, str | None] = {}
        self.interactive_without_hook: list[str] = []
        self.script_srcs: list[str] = []
        self.inline_scripts: list[str] = []
        self._in_script = False
        self._script_has_src = False
        self._script_tag = ""
        self._script_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_l = tag.lower()
        attr_map = {k.lower(): v for k, v in attrs}
        if tag_l == "html" and not self.html_attrs:
            self.html_attrs = attr_map
        if tag_l in INTERACTIVE_TAGS:
            hook = (attr_map.get("data-adam-hook") or "").strip()
            if not hook:
                self.interactive_without_hook.append(self.get_starttag_text() or f"<{tag_l}>")
        if tag_l == "script":
            src = (attr_map.get("src") or "").strip()
            self._in_script = True
            self._script_has_src = bool(src)
            self._script_tag = self.get_starttag_text() or "<script>"
            self._script_buf = []
            if src:
                self.script_srcs.append(src)
            else:
                self.inline_scripts.append(self._script_tag)

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self._script_buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._in_script:
            if self._script_has_src and "".join(self._script_buf).strip():
                self.inline_scripts.append(f"{self._script_tag} (non-empty body)")
            self._in_script = False
            self._script_buf = []


def _can_start_regex(out: list[str]) -> bool:
    """Heuristic to tell whether '/' begins a JS regex literal rather than division."""
    j = len(out) - 1
    while j >= 0 and out[j].isspace():
        j -= 1
    if j < 0:
        return True
    ch = out[j]
    if ch in "([{,;=!&|?:+-*%^~<>":
        return True
    if ch.isalnum() or ch in "$_":
        k = j
        while k >= 0 and (out[k].isalnum() or out[k] in "$_"):
            k -= 1
        return "".join(out[k + 1 : j + 1]) in _REGEX_PREV_KEYWORDS
    return False


def _strip_js_comments(js: str) -> str:
    """Strip // and /* */ JS comments while preserving strings, templates, and regexes."""
    out: list[str] = []
    stack: list[str] = ["code"]
    i, n = 0, len(js)
    while i < n:
        state = stack[-1]
        ch = js[i]
        if state in ("single", "double"):
            quote = "'" if state == "single" else '"'
            if ch == "\\" and i + 1 < n:
                out.append(js[i : i + 2])
                i += 2
                continue
            out.append(ch)
            if ch == quote:
                stack.pop()
            i += 1
        elif state == "template":
            if ch == "\\" and i + 1 < n:
                out.append(js[i : i + 2])
                i += 2
                continue
            if ch == "$" and i + 1 < n and js[i + 1] == "{":
                out.append("${")
                stack.append("interp")
                i += 2
                continue
            out.append(ch)
            if ch == "`":
                stack.pop()
            i += 1
        else:
            if ch == "/" and i + 1 < n and js[i + 1] == "/":
                i += 2
                while i < n and js[i] != "\n":
                    i += 1
                continue
            if ch == "/" and i + 1 < n and js[i + 1] == "*":
                i += 2
                while i + 1 < n and not (js[i] == "*" and js[i + 1] == "/"):
                    i += 1
                i += 2
                out.append(" ")
                continue
            if ch == "/" and _can_start_regex(out):
                out.append(ch)
                i += 1
                in_class = False
                while i < n:
                    c = js[i]
                    if c == "\\" and i + 1 < n:
                        out.append(js[i : i + 2])
                        i += 2
                        continue
                    out.append(c)
                    if c == "[":
                        in_class = True
                    elif c == "]":
                        in_class = False
                    elif c == "/" and not in_class:
                        i += 1
                        break
                    elif c == "\n":
                        i += 1
                        break
                    i += 1
                continue
            if ch == "'":
                stack.append("single")
            elif ch == '"':
                stack.append("double")
            elif ch == "`":
                stack.append("template")
            elif ch == "{":
                stack.append("brace")
            elif ch == "}" and len(stack) > 1 and stack[-1] in ("brace", "interp"):
                stack.pop()
            out.append(ch)
            i += 1
    return "".join(out)


def _skip_quoted(js: str, i: int) -> int:
    """Advance index past a string or template literal (including nested ${...})."""
    n = len(js)
    quote = js[i]
    i += 1
    while i < n:
        ch = js[i]
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if quote == "`" and ch == "$" and i + 1 < n and js[i + 1] == "{":
            i += 2
            brace_depth = 1
            while i < n and brace_depth > 0:
                c = js[i]
                if c in "'\"`":
                    i = _skip_quoted(js, i)
                    continue
                if c == "{":
                    brace_depth += 1
                elif c == "}":
                    brace_depth -= 1
                i += 1
            continue
        i += 1
        if ch == quote:
            break
    return i


def _extract_rhs(js: str, start: int) -> str:
    """Extract a JS assignment RHS starting at `start`, respecting nesting and strings."""
    n = len(js)
    i = start
    while i < n and js[i].isspace():
        i += 1
    begin = i
    depth = 0
    while i < n:
        ch = js[i]
        if ch in "'\"`":
            i = _skip_quoted(js, i)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0:
            if ch == ";":
                break
            if ch == "\n":
                j = i + 1
                while j < n and js[j].isspace():
                    j += 1
                if j >= n or js[j] not in ".+?:":
                    break
        i += 1
    return js[begin:i].strip()


def _extract_call_args(js: str, open_paren: int) -> list[str]:
    """Extract top-level comma-separated arguments starting at `(`."""
    n = len(js)
    i = open_paren + 1
    depth = 0
    arg_start = i
    args: list[str] = []
    while i < n:
        ch = js[i]
        if ch in "'\"`":
            i = _skip_quoted(js, i)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                tail = js[arg_start:i].strip()
                if tail:
                    args.append(tail)
                break
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(js[arg_start:i].strip())
            arg_start = i + 1
        i += 1
    return args


def _split_top_level_plus(expr: str) -> list[str]:
    """Split a JS expression on top-level binary `+` operators."""
    parts: list[str] = []
    depth = 0
    start = 0
    i, n = 0, len(expr)
    while i < n:
        ch = expr[i]
        if ch in "'\"`":
            i = _skip_quoted(expr, i)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "+" and depth == 0:
            if i + 1 < n and expr[i + 1] == "+":
                i += 2
                continue
            parts.append(expr[start:i].strip())
            start = i + 1
        i += 1
    parts.append(expr[start:].strip())
    return [p for p in parts if p]


def _untagged_template_interpolations(expr: str) -> list[str]:
    """Return raw `${...}` expressions inside untagged template literals."""
    found: list[str] = []
    i, n = 0, len(expr)
    while i < n:
        ch = expr[i]
        if ch in "'\"":
            i = _skip_quoted(expr, i)
            continue
        if ch == "`":
            tagged = bool(_TAGGED_HTML_RE.search(expr[:i]))
            end = _skip_quoted(expr, i)
            if not tagged:
                j = i + 1
                while j < end - 1:
                    c = expr[j]
                    if c == "\\" and j + 1 < end - 1:
                        j += 2
                        continue
                    if c == "$" and j + 1 < end - 1 and expr[j + 1] == "{":
                        k = j + 2
                        bdepth = 1
                        while k < end - 1 and bdepth > 0:
                            cc = expr[k]
                            if cc in "'\"`":
                                k = _skip_quoted(expr, k)
                                continue
                            if cc == "{":
                                bdepth += 1
                            elif cc == "}":
                                bdepth -= 1
                            k += 1
                        found.append(expr[j + 2 : k - 1].strip())
                        j = k
                        continue
                    j += 1
            i = end
            continue
        i += 1
    return found


def _assert_safe_html_expr(expr: str, context: str) -> None:
    parts = _split_top_level_plus(expr)
    assert parts, f"empty HTML sink expression in {context}"
    for part in parts:
        assert _SAFE_HTML_PREFIX_RE.match(part), f"unsafe HTML sink in {context}: {part[:100]}"
        if part.startswith("rows.map("):
            for interp in _untagged_template_interpolations(part):
                assert _SAFE_INTERP_RE.match(interp), (
                    f"unescaped interpolation `${{{interp}}}` inside rows.map in {context}"
                )


@pytest.fixture(scope="module")
def harden_mod():
    script = ROOT / "scripts" / "harden_adam_daily.py"
    assert script.is_file(), f"missing hardening script: {script}"
    spec = importlib.util.spec_from_file_location("harden_adam_daily", script)
    assert spec is not None and spec.loader is not None, f"failed to load spec for {script}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", TERMINALS)
def test_terminal_page_is_kit_wired_and_hooked(name):
    html = (SHOWCASE / f"{name}.html").read_text(encoding="utf-8")
    inspector = _TerminalHTMLInspector()
    inspector.feed(html)
    inspector.close()

    assert "<!-- adam-kit -->" in html
    assert any("adam-kit.js" in src for src in inspector.script_srcs), f"{name}.html missing adam-kit.js script"
    assert f"js/{name}.js" in inspector.script_srcs, f"{name}.html missing js/{name}.js script"
    assert inspector.html_attrs.get("data-adam-family") == "publications"
    assert "not investment advice" in html.lower()
    assert not inspector.interactive_without_hook, (
        f"{name}.html has interactive elements without data-adam-hook: {inspector.interactive_without_hook}"
    )
    assert not inspector.inline_scripts, (
        f"terminals must not embed inline scripts / literal datasets: {inspector.inline_scripts}"
    )


@pytest.mark.parametrize("name", TERMINALS)
def test_terminal_js_reads_published_json_and_escapes(name):
    js_raw = (SHOWCASE / "js" / f"{name}.js").read_text(encoding="utf-8")
    js = _strip_js_comments(js_raw)

    assert "data/publications/" in js
    assert re.search(r"\b(?:K|AdamKit)\.publish\(", js), f"{name}.js must publish data to AdamKit"
    assert not re.search(r"\bdocument\.write\s*\(", js), f"{name}.js must not use document.write"
    assert not re.search(r"\bmarked\.parse\s*\(", js), f"{name}.js must use AdamKit.markdown instead of marked.parse"

    for m in _DOM_SINK_RE.finditer(js):
        rhs = _extract_rhs(js, m.end())
        _assert_safe_html_expr(rhs, f"{name}.js ({m.group(0).strip()})")

    for m in _INSERT_ADJACENT_RE.finditer(js):
        args = _extract_call_args(js, m.end() - 1)
        assert len(args) == 2, f"unexpected insertAdjacentHTML call in {name}.js: {args}"
        _assert_safe_html_expr(args[1], f"{name}.js insertAdjacentHTML")


def test_canonical_roles_registered():
    canonical = json.loads((SHOWCASE / "canonical.json").read_text(encoding="utf-8"))
    showcase_root = SHOWCASE.resolve()
    for name, role in TERMINALS.items():
        assert role in canonical, f"missing role {role!r} in canonical.json"
        assert canonical[role]["canonical"] == f"{name}.html"
        members = canonical[role]["members"]
        assert isinstance(members, list) and members, f"empty members list for {role!r}"
        for member in members:
            assert isinstance(member, str) and member.strip(), f"invalid member path: {member!r}"
            member_path = (SHOWCASE / member).resolve()
            assert member_path.is_relative_to(showcase_root), f"member escapes showcase/: {member!r}"
            assert member_path.is_file(), f"missing canonical member file: {member}"


def test_terminal_inputs_exist():
    for path in ("data/publications/index.json", "data/publications/price_targets_latest.json"):
        payload = json.loads((SHOWCASE / path).read_text(encoding="utf-8"))
        assert isinstance(payload, dict), f"{path} must be a JSON object"
        assert payload.get("_estimated") is True, f"{path} missing _estimated: true"


def test_adam_daily_pages_are_hardened(harden_mod):
    pages = list(harden_mod.pages())
    assert pages, "expected harden_adam_daily.pages() to yield at least one page"
    for page in pages:
        text = page.read_text(encoding="utf-8")
        assert harden_mod.unescaped_interpolations(text) == [], page
        assert not re.search(r"\bmarked\.parse\s*\(", text), page
        for name in harden_mod._markup_vars(text):
            assert f"${{AdamKit.esc({name})}}" not in text, (page, name)
        assert not re.search(r"\.(?:inner|outer)HTML\s*\+?=\s*response\b", text), page
        if re.search(r"</head>", text, re.IGNORECASE):
            assert "<!-- adam-kit -->" in text, page


def test_harden_escapes_user_input(harden_mod):
    src = (
        "<head></head><script>"
        "out.innerHTML += `bash: ${cmd} ${rowsHtml} ${items.map(i => `<b>${i}</b>`).join('')}`;"
        "</script>"
    )
    out = harden_mod.harden(src, "../kit")
    assert "<!-- adam-kit -->" in out
    assert "../kit/adam-kit.js" in out
    assert "${AdamKit.esc(cmd)}" in out
    assert "${rowsHtml}" in out
    assert "${AdamKit.esc(i)}" in out
    assert harden_mod.unescaped_interpolations(out) == []
    assert harden_mod.harden(out, "../kit") == out


def test_harden_keeps_markup_variables_and_markdown_fences(harden_mod):
    src = (
        "<head></head><script>let btns = ''; btns += `<button>${t}</button>`;"
        "const dot = on ? `<i></i>` : '';"
        "const md2 = AdamKit.markdown(x); const lst = xs.map(v => {\n const t = v; return `<b>${t}</b>`;\n}).join('');"
        "c.innerHTML = `${btns}${dot}${AdamKit.esc(btns)}${name}${AdamKit.esc(md2)}${lst}`; "
        "d.innerHTML = marked.parse(AdamKit.esc(md));"
        "e.innerHTML = marked.parse(raw);</script>"
    )
    out = harden_mod.harden(src, "../kit")
    assert "${btns}${dot}${btns}${AdamKit.esc(name)}${md2}${lst}" in out
    assert "${AdamKit.esc(t)}" in out
    assert "AdamKit.markdown(md)" in out
    assert "AdamKit.markdown(raw)" in out
    assert not re.search(r"\bmarked\.parse\s*\(", out)
    assert harden_mod.unescaped_interpolations(out) == []
    assert harden_mod.harden(out, "../kit") == out
