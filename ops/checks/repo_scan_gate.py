"""Minimal AST Repo Scanner & Sentinel P0 Gate (Repomix + PurpleLlama + Sentinel)."""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

BANNED_CALLS = frozenset(
    {
        "pickle.load",
        "pickle.loads",
        "eval",
        "exec",
        "importlib.import_module",
        "xml.etree.ElementTree.parse",
        "xml.etree.ElementTree.fromstring",
        "ssl._create_unverified_context",
    }
)
BANNED_PATH_A_UI_IMPORTS = frozenset({"streamlit", "react", "dash", "gradio"})


@dataclass(frozen=True)
class RepoScanVerdict:
    layer_0_safe: int
    loc_count: int
    sha256_digest: str
    public_signatures: tuple[str, ...]
    violations: tuple[str, ...]


def scan_python_module(path: Path, is_path_a: bool = True) -> RepoScanVerdict:
    source = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    lines = [ln for ln in source.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []
    signatures: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            signatures.append(f"{type(node).__name__}:{node.name}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and is_path_a:
            mods = [a.name.split(".")[0] for a in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module.split(".")[0])
            for m in mods:
                if m in BANNED_PATH_A_UI_IMPORTS:
                    violations.append(f"PATH_A_UI_LEAK:{m}")
        elif isinstance(node, ast.Call):
            try:
                func_name = ast.unparse(node.func)
            except Exception:
                func_name = ""
            if func_name in BANNED_CALLS:
                violations.append(f"SENTINEL_P0_BAN:{func_name}")

    layer_0 = 1 if (not violations and len(lines) <= 1500) else 0
    return RepoScanVerdict(
        layer_0_safe=layer_0,
        loc_count=len(lines),
        sha256_digest=digest,
        public_signatures=tuple(signatures),
        violations=tuple(violations),
    )
