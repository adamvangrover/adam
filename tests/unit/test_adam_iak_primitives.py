"""
Unit tests for ADAM Intent & Attention Kernel (ADAM-IAK-CORE Spec 1.0.0) primitives.
"""

import math
import pytest
import torch
from src.iak.adam_iak_primitives import (
    DtypeSafeMaskedEntmaxSoftmax,
    UncertaintyGatedCRFSlotDecoder,
    DeterministicPolicyEpistemicGate,
    bisect_threshold_entmax15,
)


def test_masked_entmax_softmax_dtypes_and_modes():
    """Verify FP32, FP16, and BF16 handling across all pooling modes without NaN/overflow."""
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    modes = ["softmax", "sparsemax", "1.5-entmax", "1.5-relu"]

    for dtype in dtypes:
        for mode in modes:
            pooling = DtypeSafeMaskedEntmaxSoftmax(mode=mode, normalize_scale=True)
            logits = torch.tensor([[1.0, 2.5, -0.5, 4.0], [0.1, -10.0, 0.5, 0.2]], dtype=dtype)
            mask = torch.tensor([[True, True, True, True], [True, False, True, True]])

            probs = pooling(logits, mask=mask)

            assert probs.dtype == dtype
            assert not torch.isnan(probs).any()
            assert not torch.isinf(probs).any()
            # Masked position (0, 1) in batch row 1 should be 0
            assert probs[1, 1].item() == 0.0
            # Sum of valid positions should equal 1.0 (with tolerance matching dtype precision)
            row0_sum = probs[0].sum().item()
            row1_sum = probs[1].sum().item()
            tol = 1e-2 if dtype == torch.bfloat16 else 1e-3
            assert abs(row0_sum - 1.0) < tol
            assert abs(row1_sum - 1.0) < tol


def test_masked_entmax_softmax_fully_masked_row():
    """Verify that a fully masked row emits exact 0.0 vector instead of NaN or uniform noise."""
    pooling = DtypeSafeMaskedEntmaxSoftmax(mode="softmax", normalize_scale=False)
    logits = torch.tensor([[2.0, 1.0, 0.5], [3.0, 4.0, 5.0]], dtype=torch.float32)
    # Row 1 is fully masked
    mask = torch.tensor([[True, True, True], [False, False, False]])

    probs = pooling(logits, mask=mask)

    assert not torch.isnan(probs).any()
    # Row 0 sum should be ~1.0
    assert abs(probs[0].sum().item() - 1.0) < 1e-4
    # Row 1 should be all 0.0
    assert torch.all(probs[1] == 0.0)


def test_uncertainty_gated_crf_slot_decoder_energy_ood():
    """Verify Helmholtz free-energy OOD detection and null intent gating."""
    num_slots = 5
    intent_dim = 16
    slot_hidden_dim = 32

    decoder = UncertaintyGatedCRFSlotDecoder(
        num_slots=num_slots,
        intent_dim=intent_dim,
        slot_hidden_dim=slot_hidden_dim,
        energy_threshold=2.0,
        temperature=1.0
    )

    batch_size = 2
    seq_len = 4
    slot_hidden = torch.randn(batch_size, seq_len, slot_hidden_dim)
    intent_embed = torch.randn(batch_size, intent_dim)

    # In-distribution logits (high max logit -> low energy)
    id_logits = torch.tensor([[10.0, 0.1, 0.2], [12.0, -1.0, 0.5]])
    res_id = decoder(slot_hidden, id_logits, intent_embed)

    assert not res_id["is_ood"][0].item()
    assert not res_id["is_ood"][1].item()
    assert len(res_id["best_paths"]) == batch_size
    assert len(res_id["best_paths"][0]) == seq_len

    # Out-of-distribution logits (low max logit / negative logits -> high energy > 2.0)
    ood_logits = torch.tensor([[-5.0, -6.0, -7.0], [-10.0, -10.0, -10.0]])
    res_ood = decoder(slot_hidden, ood_logits, intent_embed)

    assert res_ood["is_ood"][0].item()
    assert res_ood["is_ood"][1].item()
    assert res_ood["gated_intent_used"][0].item()


def test_crf_decoder_constrained_viterbi_decoding():
    """Verify that illegal BIO state transitions are strictly masked during decoding."""
    # 3 slots: 0=O, 1=B-PER, 2=I-PER
    # Illegal transition: O (0) -> I-PER (2)
    illegal_transitions = [(0, 2)]

    decoder = UncertaintyGatedCRFSlotDecoder(
        num_slots=3,
        intent_dim=8,
        slot_hidden_dim=16,
        energy_threshold=100.0,
        illegal_transitions=illegal_transitions
    )

    slot_hidden = torch.randn(1, 3, 16)
    intent_logits = torch.tensor([[5.0, 0.0]])
    intent_embed = torch.randn(1, 8)

    output = decoder(slot_hidden, intent_logits, intent_embed)
    best_path = output["best_paths"][0]

    # Verify no illegal transition O (0) -> I-PER (2) occurred in path
    for t in range(len(best_path) - 1):
        from_tag, to_tag = best_path[t], best_path[t + 1]
        assert not (from_tag == 0 and to_tag == 2), f"Illegal transition {from_tag}->{to_tag} detected!"


def test_deterministic_policy_epistemic_gate_valid_and_rejection():
    """Verify policy gate epistemic validation, missing financial input rejection, and envelope format."""
    gate = DeterministicPolicyEpistemicGate(
        conviction_threshold_autonomous=0.85,
        conviction_threshold_hitl=0.50
    )

    # 1. Successful Autonomous Approval
    slots = {"ebitda": 1500000.0, "leverage_ratio": 2.5}
    intent_pred = {
        "intents": ["credit_underwriting"],
        "confidence": 0.92,
        "epistemic_status": "MODEL_EST",
        "is_ood": False
    }

    env1 = gate.evaluate(
        extracted_slots=slots,
        intent_prediction=intent_pred,
        policy_rules={},
        required_financial_inputs=["ebitda", "leverage_ratio"]
    )

    assert env1["spec_version"] == "1.0.0"
    assert env1["status"] == "APPROVED"
    assert env1["slots"]["action_decision"] == "AUTONOMOUS_EXECUTION"
    assert "MISSING_FINANCIAL_INPUT_EBITDA" not in env1["validation"]["failed"]

    # 2. Rejection due to missing financial parameter (Zero-Default Prohibition)
    missing_slots = {"leverage_ratio": 2.5}
    env2 = gate.evaluate(
        extracted_slots=missing_slots,
        intent_prediction=intent_pred,
        policy_rules={},
        required_financial_inputs=["ebitda", "leverage_ratio"]
    )

    assert env2["status"] == "REJECTED"
    assert env2["slots"]["action_decision"] == "HARD_ABORT"
    assert "MISSING_FINANCIAL_INPUT_EBITDA" in env2["validation"]["failed"]

    # 3. Pending Human-in-the-Loop review (Confidence between 0.50 and 0.85)
    hitl_intent_pred = {
        "intents": ["credit_underwriting"],
        "confidence": 0.65,
        "epistemic_status": "MODEL_EST",
        "is_ood": False
    }
    env3 = gate.evaluate(
        extracted_slots=slots,
        intent_prediction=hitl_intent_pred,
        policy_rules={},
        required_financial_inputs=["ebitda", "leverage_ratio"]
    )

    assert env3["status"] == "PENDING"
    assert env3["slots"]["action_decision"] == "HUMAN_IN_THE_LOOP_REQUIRED"
    assert "HITL_REVIEW_REQUIRED" in env3["validation"]["pending"]


def test_entmax15_large_dimension():
    """Verify 1.5-entmax threshold root finding with large dimension D=1000."""
    logits = torch.randn(2, 1000)
    probs = bisect_threshold_entmax15(logits, alpha=1.5)

    assert probs.shape == (2, 1000)
    assert not torch.isnan(probs).any()
    assert (probs >= 0.0).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-4)


def test_masked_entmax_softmax_single_element_and_temperature():
    """Verify single element sequence dimension std scaling and temperature scaling."""
    # Single element dimension
    pooling_single = DtypeSafeMaskedEntmaxSoftmax(mode="softmax", normalize_scale=True)
    logits_single = torch.tensor([[3.5]], dtype=torch.float32)
    probs_single = pooling_single(logits_single)
    assert not torch.isnan(probs_single).any()
    assert abs(probs_single.item() - 1.0) < 1e-4

    # Temperature scaling
    pooling_temp = DtypeSafeMaskedEntmaxSoftmax(mode="softmax", normalize_scale=False, temperature=2.0)
    logits = torch.tensor([[0.0, 2.0]], dtype=torch.float32)
    probs_temp = pooling_temp(logits)
    expected_temp = torch.softmax(logits / 2.0, dim=-1)
    assert torch.allclose(probs_temp, expected_temp, atol=1e-4)


def test_crf_decoder_path_scores():
    """Verify UncertaintyGatedCRFSlotDecoder returns valid best_scores along with best_paths."""
    decoder = UncertaintyGatedCRFSlotDecoder(
        num_slots=4,
        intent_dim=8,
        slot_hidden_dim=16,
        energy_threshold=100.0
    )
    slot_hidden = torch.randn(2, 5, 16)
    intent_logits = torch.tensor([[10.0, 0.0], [8.0, 1.0]])
    intent_embed = torch.randn(2, 8)

    output = decoder(slot_hidden, intent_logits, intent_embed)

    assert "best_scores" in output
    assert len(output["best_scores"]) == 2
    assert isinstance(output["best_scores"][0], float)
    assert isinstance(output["best_scores"][1], float)


def test_epistemic_gate_negative_financial_inputs():
    """Verify DeterministicPolicyEpistemicGate rejects negative financial values when constrained."""
    gate = DeterministicPolicyEpistemicGate()
    slots = {"revenue": -500.0, "leverage_ratio": 2.0}
    intent_pred = {
        "intents": ["underwriting"],
        "confidence": 0.95,
        "epistemic_status": "MODEL_EST",
        "is_ood": False
    }

    env = gate.evaluate(
        extracted_slots=slots,
        intent_prediction=intent_pred,
        policy_rules={},
        required_financial_inputs=["leverage_ratio"],
        non_negative_financial_inputs=["revenue"]
    )

    assert env["status"] == "REJECTED"
    assert env["slots"]["action_decision"] == "HARD_ABORT"
    assert "NEGATIVE_FINANCIAL_INPUT_REVENUE" in env["validation"]["failed"]
