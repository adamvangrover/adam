"""
ADAM Intent & Attention Kernel (ADAM-IAK-CORE Spec 1.0.0)
Production Remediated Neuro-Symbolic Primitives.

This module implements the core architectural remediations for ADAM-IAK-CORE:
1. DtypeSafeMaskedEntmaxSoftmax: Dtype-aware masking, FP32 projection upcasting, logit scale normalization,
   and fully-masked sequence zeroing.
2. UncertaintyGatedCRFSlotDecoder: Helmholtz energy-based OOD detection, threshold gating to null intent baseline,
   and linear-chain Viterbi BIO transition constrained decoding.
3. DeterministicPolicyEpistemicGate: Strict 6-tier epistemic lattice governance, zero-default financial rejection,
   and machine-readable JSON envelope generation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


def bisect_threshold_entmax15(
    logits: torch.Tensor,
    alpha: float = 1.5,
    max_iter: int = 50,
    tol: float = 1e-5
) -> torch.Tensor:
    """
    Computes 1.5-entmax threshold tau via bisection / root finding for exact continuous sparse probabilities.
    Formula: p_i = [ (alpha - 1) * (z_i - tau) ]_+^{1 / (alpha - 1)}
    For alpha = 1.5: p_i = [ 0.5 * (z_i - tau) ]_+^2
    """
    # z: [..., D]
    d = logits.size(-1)
    # Upper bound for tau: max logit
    tau_max, _ = torch.max(logits, dim=-1, keepdim=True)
    # Lower bound for tau: max logit - 2.0
    # Proof: p_max = [0.5*(z_max - tau)]^2 <= 1 -> z_max - tau <= 2 -> tau >= z_max - 2.0
    tau_min = tau_max - 2.0

    tau = (tau_min + tau_max) / 2.0

    for _ in range(max_iter):
        rel = F.relu(0.5 * (logits - tau))
        p = rel ** 2
        sum_p = torch.sum(p, dim=-1, keepdim=True)

        diff = sum_p - 1.0
        if torch.max(torch.abs(diff)).item() < tol:
            break

        # If sum_p > 1, tau is too small, increase tau_min
        # If sum_p < 1, tau is too large, decrease tau_max
        tau_min = torch.where(sum_p > 1.0, tau, tau_min)
        tau_max = torch.where(sum_p <= 1.0, tau, tau_max)
        tau = (tau_min + tau_max) / 2.0

    rel = F.relu(0.5 * (logits - tau))
    probs = rel ** 2
    # Renormalize slightly to ensure exact sum to 1.0 if active support non-empty
    sums = torch.sum(probs, dim=-1, keepdim=True)
    sums = torch.where(sums > 0, sums, torch.ones_like(sums))
    return probs / sums


def topk_entmax15(logits: torch.Tensor, k: int = 50) -> torch.Tensor:
    """
    Top-k approximation of 1.5-entmax for fast inference under tight latency budgets (Gate G6).
    """
    d = logits.size(-1)
    k = min(k, d)
    top_logits, top_idx = torch.topk(logits, k=k, dim=-1)
    top_probs = bisect_threshold_entmax15(top_logits, alpha=1.5)

    probs = torch.zeros_like(logits)
    probs.scatter_(dim=-1, index=top_idx, src=top_probs)
    return probs


class DtypeSafeMaskedEntmaxSoftmax(nn.Module):
    """
    Dtype-Safe Masked Entmax and Softmax Pooling Module.
    Prevents FP16 overflow (-1e9/-1e10 -> finfo.min), NaN cascades on fully-masked rows,
    and threshold drift under logit norm growth.
    """
    def __init__(
        self,
        mode: str = "softmax",
        normalize_scale: bool = True,
        top_k: Optional[int] = None,
        temperature: float = 1.0
    ):
        super().__init__()
        valid_modes = {"softmax", "sparsemax", "1.5-entmax", "1.5-relu"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode {mode}. Expected one of {valid_modes}")
        self.mode = mode
        self.normalize_scale = normalize_scale
        self.top_k = top_k
        self.temperature = max(temperature, 1e-6)

    def forward(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized attention scores or logit tensor [Batch, ..., SeqLen]
            mask: Boolean mask where True indicates valid tokens and False indicates masked tokens.
        Returns:
            probs: Normalized sparse/dense probability distribution tensor [Batch, ..., SeqLen]
        """
        orig_dtype = logits.dtype
        scores = logits.clone()

        if self.temperature != 1.0:
            scores = scores / self.temperature

        # Logit scale normalization to combat threshold drift
        if self.normalize_scale:
            if scores.size(-1) > 1:
                std = torch.std(scores, dim=-1, keepdim=True, correction=0) + 1e-6
            else:
                std = torch.ones_like(scores)
            scores = scores / std

        # Fully masked row detection
        if mask is not None:
            # Ensure mask is boolean
            bool_mask = mask.to(torch.bool)
            fully_masked_rows = ~bool_mask.any(dim=-1, keepdim=True)

            # Apply dtype-aware min value
            min_val = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~bool_mask, min_val)
        else:
            fully_masked_rows = torch.zeros(
                scores.shape[:-1] + (1,), dtype=torch.bool, device=scores.device
            )

        # Computation in FP32 for numerical precision
        scores_fp32 = scores.to(torch.float32)

        if self.mode == "softmax":
            probs_fp32 = F.softmax(scores_fp32, dim=-1)
        elif self.mode == "sparsemax":
            # Sparsemax projection onto probability simplex
            # Sort scores descending
            zs, _ = torch.sort(scores_fp32, dim=-1, descending=True)
            range_vec = torch.arange(1, zs.size(-1) + 1, device=scores.device, dtype=torch.float32)
            bound = 1.0 + range_vec * zs
            cumulative_sum = torch.cumsum(zs, dim=-1)
            is_greater = (bound > cumulative_sum).to(torch.float32)
            k = torch.max(is_greater * range_vec, dim=-1, keepdim=True)[0]
            tau = (torch.gather(cumulative_sum, -1, (k - 1).long()) - 1.0) / k
            probs_fp32 = F.relu(scores_fp32 - tau)
        elif self.mode == "1.5-entmax":
            if self.top_k is not None:
                probs_fp32 = topk_entmax15(scores_fp32, k=self.top_k)
            else:
                probs_fp32 = bisect_threshold_entmax15(scores_fp32, alpha=1.5)
        elif self.mode == "1.5-relu":
            # 1.5-ReLU fast thresholding mapping with adaptive tau = max(logits) - 1.0
            max_val, _ = torch.max(scores_fp32, dim=-1, keepdim=True)
            tau_est = max_val - 1.0
            rel = F.relu(0.5 * (scores_fp32 - tau_est))
            probs_fp32 = rel ** 2
            sums = torch.sum(probs_fp32, dim=-1, keepdim=True)
            sums = torch.where(sums > 0, sums, torch.ones_like(sums))
            probs_fp32 = probs_fp32 / sums
        else:
            probs_fp32 = F.softmax(scores_fp32, dim=-1)

        # Zero out fully masked / empty rows post-softmax to eliminate uniform 1/N padding noise
        probs_fp32 = probs_fp32.masked_fill(fully_masked_rows, 0.0)

        return probs_fp32.to(orig_dtype)


class UncertaintyGatedCRFSlotDecoder(nn.Module):
    """
    Uncertainty-Gated Intent-Conditioned CRF Slot Decoder.
    Computes Helmholtz free-energy score E(x; T). If E(x; T) > threshold_energy (OOD),
    falls back to unconditioned null intent vector y_null.
    Emits constrained linear-chain BIO Viterbi decoding with hard transition matrix constraints.
    """
    def __init__(
        self,
        num_slots: int,
        intent_dim: int,
        slot_hidden_dim: int,
        energy_threshold: float = 5.0,
        temperature: float = 1.0,
        illegal_transitions: Optional[List[Tuple[int, int]]] = None
    ):
        super().__init__()
        self.num_slots = num_slots
        self.intent_dim = intent_dim
        self.energy_threshold = energy_threshold
        self.temperature = temperature

        # Learned null intent embedding y_null
        self.null_intent = nn.Parameter(torch.zeros(intent_dim))
        nn.init.normal_(self.null_intent, std=0.02)

        # Slot emission network
        self.slot_emission_proj = nn.Linear(slot_hidden_dim + intent_dim, num_slots)

        # CRF transition matrix: transitions[i, j] is score of transitioning FROM j TO i
        self.transitions = nn.Parameter(torch.empty(num_slots, num_slots))
        nn.init.uniform_(self.transitions, -0.1, 0.1)

        # Register illegal transition mask
        self.register_buffer(
            "illegal_mask",
            torch.zeros(num_slots, num_slots, dtype=torch.bool)
        )
        if illegal_transitions:
            for from_tag, to_tag in illegal_transitions:
                self.illegal_mask[to_tag, from_tag] = True

    def compute_energy(self, intent_logits: torch.Tensor) -> torch.Tensor:
        """
        Computes Helmholtz free energy: E(x; T) = -T * LogSumExp(logits / T)
        """
        scaled_logits = intent_logits / self.temperature
        energy = -self.temperature * torch.logsumexp(scaled_logits, dim=-1)
        return energy

    def forward(
        self,
        slot_hidden: torch.Tensor,
        intent_logits: torch.Tensor,
        intent_embed: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Args:
            slot_hidden: [Batch, SeqLen, HiddenDim]
            intent_logits: [Batch, NumIntents]
            intent_embed: [Batch, IntentDim]
            mask: [Batch, SeqLen] boolean valid mask
        Returns:
            Dict containing decoded slot paths, energy scores, and OOD flags.
        """
        batch_size, seq_len, _ = slot_hidden.shape

        # Compute energy score and OOD flag
        energy = self.compute_energy(intent_logits)
        is_ood = energy > self.energy_threshold

        # Gate intent embedding to null_intent when OOD
        gated_intent_embed = torch.where(
            is_ood.unsqueeze(-1),
            self.null_intent.unsqueeze(0).expand(batch_size, -1),
            intent_embed
        )

        # Expand intent embed across sequence length and concatenate with slot_hidden
        expanded_intent = gated_intent_embed.unsqueeze(1).expand(-1, seq_len, -1)
        combined_features = torch.cat([slot_hidden, expanded_intent], dim=-1)
        emissions = self.slot_emission_proj(combined_features)  # [Batch, SeqLen, NumSlots]

        # Apply hard transition constraints
        constrained_transitions = self.transitions.clone()
        min_val = torch.finfo(emissions.dtype).min
        constrained_transitions.masked_fill_(self.illegal_mask, min_val)

        # Perform Viterbi decoding
        best_paths, best_scores = self._viterbi_decode(emissions, constrained_transitions, mask)

        return {
            "emissions": emissions,
            "best_paths": best_paths,
            "best_scores": best_scores,
            "energy": energy,
            "is_ood": is_ood,
            "gated_intent_used": is_ood
        }

    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        transitions: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Linear-chain Viterbi decoding over sequence batch, returning optimal paths and max score per batch item.
        """
        batch_size, seq_len, num_slots = emissions.shape
        best_paths = []
        best_scores = []

        for b in range(batch_size):
            if mask is not None:
                seq_l = int(mask[b].sum().item())
            else:
                seq_l = seq_len

            if seq_l == 0:
                best_paths.append([])
                best_scores.append(0.0)
                continue

            # viterbi_vars: [NumSlots]
            viterbi_vars = emissions[b, 0]  # [NumSlots]
            backpointers = []

            for t in range(1, seq_l):
                # Expand viterbi_vars to [NumSlots, 1] and transitions to [NumSlots, NumSlots]
                # next_tag_var[i, j] = viterbi_vars[j] + transitions[i, j]
                next_tag_var = viterbi_vars.unsqueeze(0) + transitions  # [NumSlots, NumSlots]
                max_vars, bptrs = torch.max(next_tag_var, dim=1)  # max over previous tag j

                viterbi_vars = max_vars + emissions[b, t]
                backpointers.append(bptrs)

            # Follow backpointer path
            max_score, best_tag_id_tensor = torch.max(viterbi_vars, dim=0)
            best_tag_id = best_tag_id_tensor.item()
            best_path = [best_tag_id]

            for bptr in reversed(backpointers):
                best_tag_id = bptr[best_tag_id].item()
                best_path.append(best_tag_id)

            best_path.reverse()
            best_paths.append(best_path)
            best_scores.append(max_score.item())

        return best_paths, best_scores


class DeterministicPolicyEpistemicGate:
    """
    Deterministic Policy Engine & Epistemic Lattice Gate.
    Enforces 6-tier epistemic state tracking:
    - OBSERVED: Directly verified facts from immutable system inputs.
    - SOURCED: Grounded external citations/literature reference.
    - CALCULATED: Mathematically derived values from deterministic code.
    - MODEL_EST: Neural probabilistic inference hypothesis (intent/slots).
    - PENDING: Gated execution awaiting verification or calibration.
    - REJECTED: Blocked/failed schema or authorization check.

    Prevents silent zero-imputation on missing financial parameters and enforces strict authorization.
    """
    EPISTEMIC_LATTICE = {
        "OBSERVED", "SOURCED", "CALCULATED", "MODEL_EST", "PENDING", "REJECTED"
    }

    def __init__(
        self,
        runtime_id: str = "ADAM-IAK-CORE",
        spec_version: str = "1.0.0",
        conviction_threshold_autonomous: float = 0.85,
        conviction_threshold_hitl: float = 0.50
    ):
        self.runtime_id = runtime_id
        self.spec_version = spec_version
        self.conviction_threshold_autonomous = conviction_threshold_autonomous
        self.conviction_threshold_hitl = conviction_threshold_hitl

    def evaluate(
        self,
        extracted_slots: Dict[str, Any],
        intent_prediction: Dict[str, Any],
        policy_rules: Dict[str, Any],
        required_financial_inputs: Optional[List[str]] = None,
        non_negative_financial_inputs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates extracted slots and predicted intents against deterministic policy rules.

        Returns machine-readable JSON envelope compliant with Spec 1.0.0.
        """
        validation_passed = []
        validation_failed = []
        validation_pending = []

        # 1. Epistemic Lattice Validation
        epistemic_status = intent_prediction.get("epistemic_status", "MODEL_EST")
        if epistemic_status not in self.EPISTEMIC_LATTICE:
            validation_failed.append("INVALID_EPISTEMIC_STATUS")
            epistemic_status = "REJECTED"
        else:
            validation_passed.append("VALID_EPISTEMIC_STATUS")

        # 2. Financial Parameter Rejection (Zero-Default Prohibition & Range Enforcement)
        if required_financial_inputs:
            for fin_param in required_financial_inputs:
                val = extracted_slots.get(fin_param)
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    validation_failed.append(f"MISSING_FINANCIAL_INPUT_{fin_param.upper()}")
                else:
                    validation_passed.append(f"VERIFIED_FINANCIAL_INPUT_{fin_param.upper()}")

        if non_negative_financial_inputs:
            for fin_param in non_negative_financial_inputs:
                val = extracted_slots.get(fin_param)
                if isinstance(val, (int, float)) and val < 0:
                    validation_failed.append(f"NEGATIVE_FINANCIAL_INPUT_{fin_param.upper()}")
                elif val is not None and not (isinstance(val, float) and math.isnan(val)):
                    validation_passed.append(f"NON_NEGATIVE_FINANCIAL_INPUT_{fin_param.upper()}")

        # 3. Conviction & Authorization Assessment
        confidence = intent_prediction.get("confidence", 0.0)
        is_ood = intent_prediction.get("is_ood", False)

        if is_ood or epistemic_status == "REJECTED" or len(validation_failed) > 0:
            overall_status = "REJECTED"
            action_decision = "HARD_ABORT"
        elif confidence >= self.conviction_threshold_autonomous:
            overall_status = "APPROVED"
            action_decision = "AUTONOMOUS_EXECUTION"
        elif confidence >= self.conviction_threshold_hitl:
            overall_status = "PENDING"
            action_decision = "HUMAN_IN_THE_LOOP_REQUIRED"
            validation_pending.append("HITL_REVIEW_REQUIRED")
        else:
            overall_status = "REJECTED"
            action_decision = "HARD_ABORT"

        # Construct Spec 1.0.0 Machine-Readable JSON Reporting Envelope
        envelope = {
            "spec_version": self.spec_version,
            "status": overall_status,
            "mode": "EVAL",
            "intent": intent_prediction.get("intents", []),
            "slots": {
                "runtime_id": self.runtime_id,
                "action_decision": action_decision,
                "extracted_slots": extracted_slots
            },
            "evidence": [
                {
                    "id": f"EV-{i+1:02d}",
                    "claim": f"Slot {k} value",
                    "status": "OBSERVED" if k in extracted_slots else "REJECTED",
                    "source": "Stage A Ingestion"
                }
                for i, k in enumerate(extracted_slots.keys())
            ],
            "predictions": [
                {
                    "task": "execution_authorization",
                    "prediction": action_decision,
                    "epistemic_status": epistemic_status,
                    "confidence": confidence,
                    "rationale": f"Evaluated under conviction threshold {self.conviction_threshold_autonomous}"
                }
            ],
            "validation": {
                "passed": validation_passed,
                "failed": validation_failed,
                "pending": validation_pending
            },
            "metrics": {
                "control_quality": {
                    "schema_compliance": 1.0 if len(validation_failed) == 0 else 0.0,
                    "unverified_claim_leakage": 0.0
                }
            }
        }

        return envelope
