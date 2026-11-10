"""Property and Invariant Test Suite for ADAM-PORT-CORE Ports."""
from pathlib import Path
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from ops.checks.repo_scan_gate import FaissL1SignatureIndex, scan_python_module
from src.domains.covenant_solver import solve_smt_covenant_feasibility
from src.domains.neuro_symbolic_ports import (
    GroundedSpan,
    JevSystem1Judgment,
    cbf_safety_filter_order_size,
)
from src.iak.s1_triage_extract import (
    build_llguidance_grammar_spec,
    extract_grounded_spans,
    get_jev_json_schema,
    triage_inference,
)
from src.pdil.middleware import RustPDILBridgeGatekeeper


def test_repo_scan_gate_passes_clean_file(tmp_path: Path) -> None:
    clean_code = """
def compute_sum(a: int, b: int) -> int:
    return a + b
"""
    clean_file = tmp_path / "clean_mod.py"
    clean_file.write_text(clean_code, encoding="utf-8")

    verdict = scan_python_module(clean_file, is_path_a=True)
    assert verdict.layer_0_safe == 1
    assert len(verdict.violations) == 0
    assert "FunctionDef:compute_sum" in verdict.public_signatures


def test_repo_scan_gate_blocks_sentinel_p0_banned_calls(tmp_path: Path) -> None:
    unsafe_code = """
import pickle

def unsafe_loader(data):
    return pickle.loads(data)
"""
    unsafe_file = tmp_path / "unsafe_mod.py"
    unsafe_file.write_text(unsafe_code, encoding="utf-8")

    verdict = scan_python_module(unsafe_file, is_path_a=True)
    assert verdict.layer_0_safe == 0
    assert any("SENTINEL_P0_BAN:pickle.loads" in v for v in verdict.violations)


def test_repo_scan_gate_blocks_path_a_ui_imports(tmp_path: Path) -> None:
    ui_leak_code = """
import streamlit as st

def show_app():
    st.write("hello")
"""
    ui_file = tmp_path / "ui_mod.py"
    ui_file.write_text(ui_leak_code, encoding="utf-8")

    verdict = scan_python_module(ui_file, is_path_a=True)
    assert verdict.layer_0_safe == 0
    assert any("PATH_A_UI_LEAK:streamlit" in v for v in verdict.violations)


def test_jev_tripwire_demotes_low_conviction_or_ungrounded() -> None:
    span = GroundedSpan(
        source_doc_id="DOC_10K_2025",
        char_start=10,
        char_end=22,
        verbatim_text="EBITDA $50M",
        tier="VERIFIED",
    )

    # Valid judgment >= 0.85 with grounded span
    valid_j = JevSystem1Judgment(
        intent_choice="UNDERWRITE_CREDIT",
        conviction_score=0.92,
        next_step_metadata={"route": "direct"},
        grounded_spans=(span,),
    ).enforce_tripwire()
    assert valid_j.intent_choice == "UNDERWRITE_CREDIT"

    # Low conviction (< 0.85) -> demotes to NULL_INTENT
    low_conv_j = JevSystem1Judgment(
        intent_choice="UNDERWRITE_CREDIT",
        conviction_score=0.75,
        next_step_metadata={},
        grounded_spans=(span,),
    ).enforce_tripwire()
    assert low_conv_j.intent_choice == "NULL_INTENT"
    assert low_conv_j.next_step_metadata["next_step"] == "human_escalate"

    # Missing grounded spans -> demotes to NULL_INTENT
    ungrounded_j = JevSystem1Judgment(
        intent_choice="UNDERWRITE_CREDIT",
        conviction_score=0.95,
        next_step_metadata={},
        grounded_spans=(),
    ).enforce_tripwire()
    assert ungrounded_j.intent_choice == "NULL_INTENT"


def test_extract_grounded_spans_and_triage_helper() -> None:
    text = "The company reported EBITDA of $100 million and maintained minimum covenant DSCR."
    spans = extract_grounded_spans("DOC_001", text)
    assert len(spans) >= 2
    for s in spans:
        assert s.verbatim_text in text
        assert text[s.char_start:s.char_end] == s.verbatim_text

    res = triage_inference("UNDERWRITE_CREDIT", 0.90, spans)
    assert res.intent_choice == "UNDERWRITE_CREDIT"


def test_smt_covenant_feasibility_solver() -> None:
    # 100 EBITDA + 20 add-back = 120 EBITDAR; debt service = 40 + 50 = 90; DSCR = 120/90 = 1.33x >= 1.05
    is_sat, dscr = solve_smt_covenant_feasibility(
        base_ebitda=100.0,
        max_addback_pct=0.30,  # 30% capped at 20%
        interest=40.0,
        amortization=50.0,
        min_dscr=1.05,
    )
    assert is_sat is True
    assert np.isclose(dscr, 120.0 / 90.0)

    # Infeasible debt service
    is_sat_fail, dscr_fail = solve_smt_covenant_feasibility(
        base_ebitda=100.0,
        max_addback_pct=0.10,
        interest=80.0,
        amortization=80.0,
        min_dscr=1.05,
    )
    assert is_sat_fail is False
    assert dscr_fail < 1.05


def test_rust_pdil_bridge_gatekeeper() -> None:
    bridge = RustPDILBridgeGatekeeper(min_conviction=0.85, max_vpin=0.82, min_dscr=1.05)
    payload = b'{"deal":"PORT_ACQUISITION","val":123}'

    # Normal pass -> COMMIT_CANDIDATE
    l0, score, sha, route = bridge.evaluate(
        raw_json_bytes=payload,
        jev_conviction=0.90,
        vpin_toxicity=0.20,
        stressed_dscr=1.50,
        cedar_forbid_triggered=False,
        raw_quality_score=4.8,
    )
    assert l0 == 1
    assert score == 4.8
    assert len(sha) == 64
    assert route == "COMMIT_CANDIDATE"

    # Cedar forbid -> BLOCK_CEDAR_FORBID
    l0_cedar, _, _, route_cedar = bridge.evaluate(
        raw_json_bytes=payload,
        jev_conviction=0.90,
        vpin_toxicity=0.20,
        stressed_dscr=1.50,
        cedar_forbid_triggered=True,
        raw_quality_score=4.8,
    )
    assert l0_cedar == 0
    assert route_cedar == "BLOCK_CEDAR_FORBID"


def test_faiss_l1_signature_index() -> None:
    index = FaissL1SignatureIndex(dimension=32)
    sigs = [
        "FunctionDef:underwrite_loan",
        "FunctionDef:calculate_var",
        "ClassDef:CreditAgreement",
    ]
    added = index.add_signatures(sigs)
    assert added == 3

    results = index.search("FunctionDef:underwrite_loan", top_k=2)
    assert len(results) == 2
    # Exact match should have 0 distance (or lowest distance)
    assert results[0][0] == "FunctionDef:underwrite_loan"
    assert np.isclose(results[0][1], 0.0, atol=1e-5)


def test_llguidance_grammar_spec_and_json_schema() -> None:
    schema = get_jev_json_schema()
    assert "properties" in schema
    assert "intent_choice" in schema["properties"]
    assert "conviction_score" in schema["properties"]

    spec = build_llguidance_grammar_spec()
    assert spec["type"] == "json_schema"
    assert spec["schema"] == schema
    assert "llguidance_available" in spec


# Property-Based Testing
@given(
    vpin=st.floats(min_value=0.0, max_value=1.5),
    nominal_qty=st.floats(min_value=1.0, max_value=10000.0),
)
@settings(max_examples=50)
def test_hypothesis_cbf_safety_filter_invariant(vpin: float, nominal_qty: float) -> None:
    """Hypothesis invariant: Whenever VPIN >= 0.82, Layer 0 must clamp order quantity to exactly 0.0."""
    l0_safe, safe_qty = cbf_safety_filter_order_size(nominal_qty, vpin, vpin_limit=0.82)
    if vpin >= 0.82:
        assert l0_safe == 0
        assert safe_qty == 0.0
    else:
        assert l0_safe == 1
        assert 0.0 <= safe_qty <= nominal_qty
