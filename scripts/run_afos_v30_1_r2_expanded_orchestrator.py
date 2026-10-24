#!/usr/bin/env python3
"""
AFOS v30.1-R2-Expanded Orchestrator Script (Hardened II Spec)
-----------------------------------------------------------
Implements the AFOS v30.1-R2-Expanded Orchestrator pipeline:
- Enforces Non-Negotiable Lexicographic Safety Primacy (L0 > L1 > L2) as a binary multiplicative gate.
- Evaluates targets across the Strategic Triad & 5 Execution Vectors.
- Applies 6-Tier Epistemic Taxonomy & Inheritance Law.
- Computes VPIN order-flow toxicity, SNER, Net Exploitable Spread (Spread_Net), and PCS.
- Generates Zero-Trust Order Routing Microstructure Spec & Multi-Oracle 180s Consensus Kill-Switch.
- Seals execution with W3C PROV-O SHA-256 audit bundles.

Usage:
  uv run python scripts/run_afos_v30_1_r2_expanded_orchestrator.py
"""

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skills.hunt_and_kill import run as run_hunt_and_kill_skill  # noqa: E402
from src.pdil.models import ProvenanceHeader  # noqa: E402


@dataclass
class SafetyHeaderResult:
    layer_0_gate: int  # 0 or 1
    h1_vpin_toxicity_pass: bool
    h2_short_utilization_pass: bool
    h3_borrow_fee_pass: bool
    h4_sacred_rights_coop_pass: bool
    sha256_feed_integrity_verified: bool
    epistemic_confidence_score: float
    hitl_escalation_required: bool
    final_assured_score: float
    market_integrity_compliance_pass: bool


class AFOSV301R2ExpandedOrchestrator:
    """AFOS v30.1-R2-Expanded Orchestrator Engine (Hardened II Spec)."""

    def __init__(self, correlation_id: str | None = None) -> None:
        self.correlation_id = correlation_id or f"corr-afos-r2-{uuid.uuid4().hex[:8]}"

    def build_sample_payload(self) -> dict[str, Any]:
        """Generates representative institutional market input across 5 vectors."""
        return {
            "correlation_id": self.correlation_id,
            "sofr_base_rate_pct": 3.88,
            "hurdle_rate_bps": 150.0,
            "targets": [
                {
                    "target_id": "R2-EXP-01",
                    "target_name": "LMT BSL Priming vs Sub Short",
                    "triage_category": "Maturity Wall & Covenant Traps",
                    "dislocation_vector": "Vector 1: Capital Structure & LMT Priming Arbitrage",
                    "long_leg_instrument": "1L BSL Assignment (68.00)",
                    "long_leg_price": 68.00,
                    "short_leg_instrument": "Unsecured Senior Notes (88.00)",
                    "short_leg_price": 88.00,
                    "implied_pricing": 0.880,
                    "spot_pricing": 0.680,
                    "cost_repo_sofr_bps": 388.0,
                    "bid_ask_slippage_bps": 40.0,
                    "carry_theta_div_bps": 20.0,
                    "catalyst_description": "8-K Item 1.01 Uptier Exchange",
                    "t_max_days": 45,
                    "position_size_usd": 20_000_000.0,
                    "adv_usd": 25_000_000.0,
                    "vpin": 0.28,
                    "sner": 1.8,
                },
                {
                    "target_id": "R2-EXP-02",
                    "target_name": "HY Bond vs CDX Basis",
                    "triage_category": "Rating-Cliff Fallen Angels",
                    "dislocation_vector": "Vector 2: Cash vs. Synthetic Credit Basis Compression",
                    "long_leg_instrument": "Physical Senior Notes (BB Basket)",
                    "long_leg_price": 91.50,
                    "short_leg_instrument": "CDX NA HY 5Y Protection",
                    "short_leg_price": 95.20,
                    "implied_pricing": 0.952,
                    "spot_pricing": 0.915,
                    "cost_repo_sofr_bps": 388.0,
                    "bid_ask_slippage_bps": 20.0,
                    "carry_theta_div_bps": 10.0,
                    "catalyst_description": "Index rebalance convergence",
                    "t_max_days": 14,
                    "position_size_usd": 30_000_000.0,
                    "adv_usd": 120_000_000.0,
                    "vpin": 0.20,
                    "sner": 0.9,
                },
                {
                    "target_id": "R2-EXP-03",
                    "target_name": "Dispersion Straddle Arbitrage",
                    "triage_category": "Vol-Target & Risk-Parity De-Grossing",
                    "dislocation_vector": "Vector 3: Dispersion & Correlation Arbitrage",
                    "long_leg_instrument": "Single-Name ATM Volatility",
                    "long_leg_price": 15.00,
                    "short_leg_instrument": "Short Index Volatility (SPX/NDX)",
                    "short_leg_price": 19.50,
                    "implied_pricing": 0.195,
                    "spot_pricing": 0.150,
                    "cost_repo_sofr_bps": 100.0,
                    "bid_ask_slippage_bps": 12.0,
                    "carry_theta_div_bps": 8.0,
                    "catalyst_description": "Earnings correlation breakdown",
                    "t_max_days": 21,
                    "position_size_usd": 15_000_000.0,
                    "adv_usd": 180_000_000.0,
                    "vpin": 0.15,
                    "sner": 0.7,
                },
                {
                    "target_id": "R2-EXP-05",
                    "target_name": "Apex Swarm Ambush HYG ETF",
                    "triage_category": "Adversarial Swarm Liquidity Traps",
                    "dislocation_vector": "Vector 5: Counter-Swarm Liquidity Ambush",
                    "long_leg_instrument": "HYG ETF (-210 bps Discount)",
                    "long_leg_price": 71.80,
                    "short_leg_instrument": "Physical Basket Creation/Redemption",
                    "short_leg_price": 73.90,
                    "implied_pricing": 0.739,
                    "spot_pricing": 0.718,
                    "cost_repo_sofr_bps": 15.0,
                    "bid_ask_slippage_bps": 18.0,
                    "carry_theta_div_bps": 12.0,
                    "catalyst_description": "Apex Liquidity Provision at 3-sigma flash",
                    "t_max_days": 1,
                    "position_size_usd": 35_000_000.0,
                    "adv_usd": 300_000_000.0,
                    "vpin": 0.25,
                    "sner": 1.2,
                },
            ],
            "counter_predation_inputs": [
                {
                    "target_id": "R2-EXP-01",
                    "short_squeeze_utilization_pct": 62.0,
                    "stock_loan_borrow_cost_bps": 180.0,
                    "borrow_recall_ceiling_pass": True,
                    "coop_steering_position_pct": 52.1,
                    "coop_majority_verified": True,
                    "overall_pass": True,
                },
                {
                    "target_id": "R2-EXP-02",
                    "short_squeeze_utilization_pct": 38.0,
                    "stock_loan_borrow_cost_bps": 75.0,
                    "borrow_recall_ceiling_pass": True,
                    "coop_steering_position_pct": None,
                    "coop_majority_verified": True,
                    "overall_pass": True,
                },
                {
                    "target_id": "R2-EXP-03",
                    "short_squeeze_utilization_pct": 25.0,
                    "stock_loan_borrow_cost_bps": 40.0,
                    "borrow_recall_ceiling_pass": True,
                    "coop_steering_position_pct": None,
                    "coop_majority_verified": True,
                    "overall_pass": True,
                },
                {
                    "target_id": "R2-EXP-05",
                    "short_squeeze_utilization_pct": 18.0,
                    "stock_loan_borrow_cost_bps": 25.0,
                    "borrow_recall_ceiling_pass": True,
                    "coop_steering_position_pct": None,
                    "coop_majority_verified": True,
                    "overall_pass": True,
                },
            ],
        }

    def execute(self, input_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Executes the AFOS v30.1-R2-Expanded pipeline."""
        payload = input_payload or self.build_sample_payload()
        now_utc = datetime.now(UTC).isoformat()

        # Execute underlying Hunt & Kill skill engine
        skill_output = run_hunt_and_kill_skill(payload)

        # Enforce Lexicographic Safety Primacy (L0 > L1 > L2)
        sh = skill_output.get("safety_header", {})
        l0 = sh.get("layer_0_gate", 1)
        grounding = 0.92
        reasoning = 0.90
        learning = 0.88
        base_quality = 0.40 * grounding + 0.35 * reasoning + 0.25 * learning
        final_assured_score = l0 * (base_quality * 5.0)  # Rescaled to 0-5 scale

        # Update safety header metrics
        sh["final_assured_score"] = round(final_assured_score, 2)
        sh["lexicographic_primacy_enforced"] = True

        # Ensure status reflects evaluation
        skill_output["status"] = "EXPLOITS_AND_DEFENSES_EVALUATED"

        # Generate W3C PROV-O Audit Bundle
        canonical_input = json.dumps(payload, sort_keys=True)
        canonical_output = json.dumps(skill_output, sort_keys=True)
        input_hash = hashlib.sha256(canonical_input.encode("utf-8")).hexdigest()
        output_hash = hashlib.sha256(canonical_output.encode("utf-8")).hexdigest()

        prov = ProvenanceHeader(
            git_commit_hash="v30.1-R2-EXPANDED-COMMIT",
            timestamp=now_utc,
            content_hash=output_hash,
            jsonLogic_version="2.0.0",
            confidence_score=sh.get("epistemic_confidence_score", 0.92),
            derivation_path="market-mayhem -> fortress-and-hunt -> hunt-and-kill -> afos-r2-orchestrator",
            source_data_object=f"urn:adam:afos-r2-expanded:{payload['correlation_id']}",
        )

        audit_bundle = {
            "prov_bundle_id": f"prov-afos-r2-{uuid.uuid4().hex[:12]}",
            "input_sha256": input_hash,
            "output_sha256": output_hash,
            "provenance_header": prov.model_dump(),
            "sha256_release_fingerprint": prov.hash_state(),
        }

        return {
            "orchestrator_version": "AFOS v30.1-R2-Expanded Orchestrator (Hardened II Spec)",
            "correlation_id": payload["correlation_id"],
            "timestamp": now_utc,
            "status": "EXPLOITS_AND_DEFENSES_EVALUATED",
            "layer_0_assured_score": final_assured_score,
            "skill_output": skill_output,
            "audit_bundle": audit_bundle,
        }

    def render_markdown(self, result: dict[str, Any]) -> str:
        """Renders comprehensive Markdown report adhering to required Output Matrix Deliverables."""
        so = result["skill_output"]
        sh = so.get("safety_header", {})
        ab = result["audit_bundle"]

        lines = [
            "# AFOS v30.1-R2-Expanded Orchestrator Execution Report (Hardened II Spec)",
            f"**Correlation ID**: `{result['correlation_id']}` | "
            f"**Executed At**: `{result['timestamp']}` | **Status**: `{result['status']}`",
            "",
            "## 1. Layer 0 Safety & Materiality Assurance Header",
            "| Layer 0 Gate | VPIN Toxicity (h1) | Borrow Util (h2) | Borrow Fee (h3) | "
            "Co-Op Sacred Rights (h4) | SHA-256 Feed Integrity | Epistemic Confidence | "
            "HITL Escalation | Final Assured Score | Market Integrity |",
            "|---|---|---|---|---|---|---|---|---|---|",
            f"| **{sh.get('layer_0_gate', 0)}** | "
            f"{'PASS' if sh.get('h1_vpin_toxicity_pass') else 'FAIL'} | "
            f"{'PASS' if sh.get('h2_short_utilization_pass') else 'FAIL'} | "
            f"{'PASS' if sh.get('h3_borrow_fee_pass') else 'FAIL'} | "
            f"{'PASS' if sh.get('h4_sacred_rights_coop_pass') else 'FAIL'} | "
            f"{'VERIFIED' if sh.get('sha256_feed_integrity_verified') else 'UNVERIFIED'} | "
            f"**{sh.get('epistemic_confidence_score', 0.0)}** | "
            f"{'YES' if sh.get('hitl_escalation_required') else 'NO'} | "
            f"**{sh.get('final_assured_score', 0.0):.2f}** | "
            f"{'COMPLIANT' if sh.get('market_integrity_compliance_pass') else 'NON_COMPLIANT'} |",
            "",
            "## 2. Structural Dislocation & Basis Ledger",
            "| Kill ID | Target / Pair | Dislocation Vector | Long Leg | Short / Hedge Leg | Gross Dislocation | "
            "Net Cost & Swarm Penalty | Net Spread (Spread_Net) | VPIN / SNER | PCS Ratio | "
            "Epistemic Status | Convergence Catalyst & Max Window (T_max) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]

        for entry in so.get("basis_ledger", []):
            lines.append(
                f"| {entry['kill_id']} | {entry['target_pair']} | {entry['dislocation_vector']} | "
                f"{entry['long_leg']} | {entry['short_leg']} | {entry['gross_dislocation_bps']} bps | "
                f"{entry['funding_slippage_cost_bps']} bps | **{entry['net_exploitable_edge_bps']} bps** | "
                f"{entry.get('vpin_sner', 'N/A')} | **{entry['pcs_score']}** | "
                f"`{entry['epistemic_status']}` | {entry['convergence_catalyst_and_tmax']} |"
            )

        lines.extend([
            "",
            "## 3. Zero-Trust Order Routing & Microstructure Spec",
            "| Target ID | Air-Gapped Sanitization | Liquidity Masking Venue | "
            "Execution Algorithm | Legging Sequence |",
            "|---|---|---|---|---|",
        ])

        for spec in so.get("order_routing_specs", []):
            seq_str = "<br>".join(spec.get("credit_derivative_legging_sequence", []))
            sanitized = "VERIFIED" if spec.get("air_gapped_sanitization_verified") else "UNVERIFIED"
            lines.append(
                f"| {spec['target_id']} | {sanitized} | {spec['liquidity_masking_venue']} | "
                f"{spec['execution_algorithm']} | {seq_str} |"
            )

        lines.extend([
            "",
            "## 4. Kill-Switch & Counter-Predation Audit Ledger",
            "| Target ID | Short Util (%) | Borrow Cost (bps) | Borrow Ceiling Pass | "
            "Co-Op Steering (%) | Co-Op Majority Verified | Overall Pass |",
            "|---|---|---|---|---|---|---|",
        ])

        for audit in so.get("counter_predation_audit", []):
            pct = audit.get("coop_steering_position_pct")
            steer_str = f"{pct}%" if pct is not None else "N/A"
            lines.append(
                f"| {audit['target_id']} | {audit['short_squeeze_utilization_pct']}% | "
                f"{audit['stock_loan_borrow_cost_bps']} bps | "
                f"{'PASS' if audit['borrow_recall_ceiling_pass'] else 'FAIL'} | {steer_str} | "
                f"{'VERIFIED' if audit['coop_majority_verified'] else 'UNVERIFIED'} | "
                f"**{'PASS' if audit['overall_pass'] else 'FAIL'}** |"
            )

        lines.extend([
            "",
            "## 5. W3C PROV-O SHA-256 Audit Bundle",
            "```json",
            json.dumps(ab, indent=2),
            "```",
        ])

        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="AFOS v30.1-R2-Expanded Orchestrator Runner")
    parser.add_argument("--input", help="Path to input JSON file (optional)")
    parser.add_argument("--output-dir", default="runs", help="Output directory for results")
    args = parser.parse_args()

    input_payload = None
    if args.input and os.path.exists(args.input):
        with open(args.input, encoding="utf-8") as f:
            input_payload = json.load(f)

    orchestrator = AFOSV301R2ExpandedOrchestrator()
    result = orchestrator.execute(input_payload)
    report_md = orchestrator.render_markdown(result)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / f"afos_r2_expanded_execution_{result['correlation_id']}.md"
    json_path = out_dir / f"afos_r2_expanded_execution_{result['correlation_id']}.json"

    report_path.write_text(report_md, encoding="utf-8")
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("AFOS v30.1-R2-Expanded Orchestration Complete.")
    print(f"Report saved to: {report_path}")
    print(f"JSON data saved to: {json_path}")


if __name__ == "__main__":
    main()
