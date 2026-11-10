"""Minimal AST Repo Scanner & Sentinel P0 Gate (Repomix + PurpleLlama + Sentinel)."""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import faiss  # type: ignore[import-not-found]
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

BANNED_CALLS = frozenset({
    "pickle.load",
    "pickle.loads",
    "eval",
    "exec",
    "importlib.import_module",
    "xml.etree.ElementTree.parse",
    "xml.etree.ElementTree.fromstring",
    "ssl._create_unverified_context",
})
BANNED_PATH_A_UI_IMPORTS = frozenset({"streamlit", "react", "dash", "gradio"})


@dataclass(frozen=True)
class RepoScanVerdict:
    layer_0_safe: int
    loc_count: int
    sha256_digest: str
    public_signatures: tuple[str, ...]
    violations: tuple[str, ...]


def scan_python_module(path: Path, is_path_a: bool = True) -> RepoScanVerdict:
    """Compresses candidate repos/files to AST signatures and screens for Sentinel P0 bans."""
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


class FaissL1SignatureIndex:
    """In-memory FAISS L1 vector index for repo AST signatures and code chunks."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension
        self.signatures: list[str] = []
        if HAS_FAISS:
            self.index: Any = faiss.IndexFlatL2(dimension)
        else:
            self.index = None
            self._vectors: list[list[float]] = []

    def _hash_to_embedding(self, text: str) -> list[float]:
        """Deterministic pseudo-embedding for fast L1 indexing without heavy models."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Derive float vector of size `self.dimension` from sha256 bytes
        vals = []
        for i in range(self.dimension):
            byte_val = h[i % len(h)]
            vals.append((byte_val / 255.0) * 2.0 - 1.0)
        return vals

    def add_signatures(self, signatures: list[str]) -> int:
        if not signatures:
            return 0
        vectors = [self._hash_to_embedding(s) for s in signatures]
        self.signatures.extend(signatures)
        if HAS_FAISS:
            arr = np.array(vectors, dtype=np.float32)
            self.index.add(arr)
        else:
            self._vectors.extend(vectors)
        return len(signatures)

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        if not self.signatures:
            return []
        q_vec = self._hash_to_embedding(query)
        top_k = min(top_k, len(self.signatures))
        if HAS_FAISS:
            q_arr = np.array([q_vec], dtype=np.float32)
            distances, indices = self.index.search(q_arr, top_k)
            return [(self.signatures[idx], float(dist)) for dist, idx in zip(distances[0], indices[0]) if idx >= 0]

        # Analytical fallback
        q_np = np.array(q_vec)
        dists = [float(np.sum((q_np - np.array(v)) ** 2)) for v in self._vectors]
        ranked = sorted(zip(self.signatures, dists), key=lambda x: x[1])[:top_k]
        return ranked
