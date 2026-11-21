"""Seam: first-party imports in execution code must resolve to a real module path."""
import ast
import importlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ROOTS = ("core", "services", "server", "adam_os", "src", "scripts")
SKIP_PARTS = {"experimental", "archive", "node_modules", "tests", "__pycache__"}
FIRST_PARTY = {"core", "services", "server", "adam_os", "src", "adam", "scripts"}
# Known orphans with no current implementation; remove an entry once it is fixed.
ALLOWED = {"core.analysis.xai.explainer"}


def _resolves(module: str) -> bool:
    path = REPO.joinpath(*module.split("."))
    return path.with_suffix(".py").is_file() or path.is_dir()


def _unresolved_imports():
    for root in ROOTS:
        for py in (REPO / root).rglob("*.py"):
            if SKIP_PARTS.intersection(py.relative_to(REPO).parts):
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                else:
                    continue
                for name in names:
                    if name.split(".")[0] in FIRST_PARTY and name not in ALLOWED and not _resolves(name):
                        yield f"{py.relative_to(REPO)}:{node.lineno} -> {name}"


def test_first_party_imports_resolve():
    assert list(_unresolved_imports()) == []


@pytest.mark.parametrize(
    "module, attr",
    [
        ("core.credit_sentinel.covenant_agent", "CovenantAnalystAgent"),
        ("core.agents.specialized.credit_snc", "SNCRatingAgent"),
        ("core.agents.specialized.financial_covenant_agent", "CovenantAnalystAgent"),
    ],
)
def test_repointed_modules_import(module, attr):
    assert hasattr(importlib.import_module(module), attr)


_WORKFLOW_TARGET = re.compile(r"(?:pip install -r|python3?|uv run python)\s+([\w./-]+\.(?:py|txt))")


def test_workflow_file_targets_exist():
    missing = []
    for wf in sorted((REPO / ".github" / "workflows").glob("*.y*ml")):
        for lineno, line in enumerate(wf.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for target in _WORKFLOW_TARGET.findall(line):
                if not (REPO / target).exists() and not (wf.parent / target).exists():
                    missing.append(f"{wf.name}:{lineno} -> {target}")
    assert missing == []
