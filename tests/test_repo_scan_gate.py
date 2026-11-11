"""CI Verification Suite for ops/checks/repo_scan_gate.py AST Sentinel P0 Scanner."""
from pathlib import Path
from ops.checks.repo_scan_gate import scan_python_module


def test_clean_module_passes_scan(tmp_path: Path) -> None:
    test_file = tmp_path / "clean_sample.py"
    test_file.write_text(
        "def compute_sum(a: int, b: int) -> int:\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    verdict = scan_python_module(test_file, is_path_a=True)
    assert verdict.layer_0_safe == 1
    assert len(verdict.violations) == 0
    assert "FunctionDef:compute_sum" in verdict.public_signatures


def test_sentinel_p0_banned_calls_blocked(tmp_path: Path) -> None:
    test_file = tmp_path / "unsafe_sample.py"
    test_file.write_text(
        "import pickle\n"
        "def unsafe_load(payload):\n"
        "    return pickle.loads(payload)\n",
        encoding="utf-8",
    )
    verdict = scan_python_module(test_file, is_path_a=True)
    assert verdict.layer_0_safe == 0
    assert "SENTINEL_P0_BAN:pickle.loads" in verdict.violations


def test_path_a_ui_import_leak_blocked(tmp_path: Path) -> None:
    test_file = tmp_path / "ui_leak.py"
    test_file.write_text(
        "import streamlit as st\n"
        "def render():\n"
        "    st.title('Forbidden')\n",
        encoding="utf-8",
    )
    verdict = scan_python_module(test_file, is_path_a=True)
    assert verdict.layer_0_safe == 0
    assert "PATH_A_UI_LEAK:streamlit" in verdict.violations
