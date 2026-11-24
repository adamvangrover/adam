"""Architectural boundary gates: directory ownership, Path A/B imports, unsafe calls, agent catalog."""
import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DIRMAP = json.loads((REPO / "machine_index.json").read_text())["directory_map"]["directories"]
PATH_A = sorted(d for d, t in DIRMAP.items() if t == "PATH_A")
FORBIDDEN = {d for d, t in DIRMAP.items() if t in ("PATH_B", "DEPRECATED_ARCHIVE")} | {"tinker_lab", "experimental"}
LAB_INSIDE_A = ("core/experimental/", "core/tests/")
# Legacy imports that predate the gate. Remove entries as they are fixed; never add new ones.
KNOWN_BOUNDARY_VIOLATIONS = {("core/system/system_controller.py", "agents")}
UNSAFE_CALLS = {("eval",), ("exec",), ("pickle", "load"), ("pickle", "loads"), ("dill", "load"), ("dill", "loads")}


def _path_a_files():
    for top in PATH_A:
        for py in (REPO / top).rglob("*.py"):
            rel = py.relative_to(REPO).as_posix()
            if not rel.startswith(LAB_INSIDE_A) and "node_modules" not in rel:
                yield py, rel


def _trees():
    for py, rel in _path_a_files():
        try:
            yield py, rel, ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue


def _is_local(py: Path, top: str) -> bool:
    """Script-style `import agents` next to a sibling `agents/` resolves locally, not to the repo root."""
    return (py.parent / top).exists() or (py.parent / f"{top}.py").exists()


def test_every_top_level_directory_is_tagged():
    try:
        tracked = subprocess.run(
            ["git", "ls-tree", "-d", "--name-only", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git unavailable")
    untagged = sorted(set(tracked) - set(DIRMAP))
    assert not untagged, f"Tag new top-level directories in machine_index.json directory_map: {untagged}"


def test_path_a_does_not_import_path_b():
    violations = set()
    for py, rel, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            else:
                continue
            for mod in mods:
                top = mod.split(".")[0]
                if (top in FORBIDDEN and not _is_local(py, top)) or mod.startswith("core.experimental"):
                    violations.add((rel, mod))
    assert violations - KNOWN_BOUNDARY_VIOLATIONS == set()
    assert KNOWN_BOUNDARY_VIOLATIONS <= violations, "A known violation was fixed: drop it from the allowlist."


def test_path_a_has_no_unsafe_calls():
    found = []
    for _, rel, tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            key = (f.id,) if isinstance(f, ast.Name) else (
                (f.value.id, f.attr) if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) else None
            )
            if key in UNSAFE_CALLS:
                found.append(f"{rel}:{node.lineno} {'.'.join(key)}")
    assert found == [], "Use core.security.safe_unpickler / ast.literal_eval instead: " + ", ".join(found)


def test_agent_catalog_files_exist():
    catalog = (REPO / "AGENT_CATALOG.md").read_text()
    paths = re.findall(r"\*\*File:\*\* `([^`]+\.py)`", catalog)
    assert paths, "AGENT_CATALOG.md lists no agent files"
    missing = [p for p in paths if not (REPO / p).is_file()]
    assert missing == [], f"AGENT_CATALOG.md points at missing files (mark them _planned_ or fix the path): {missing}"
