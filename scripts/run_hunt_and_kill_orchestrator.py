#!/usr/bin/env python3
"""
ADAM v30.1-R-Patch Module 8: Hunt & Kill Orchestrator Runner (Hardened Spec)
Executes the 4-stage Hunt & Kill pipeline and seals execution outputs with W3C PROV-O SHA-256 audit bundles.
Renders Layer 0 Safety Header, forced-flow triage & VPIN metrics, ADV market impact, lifecycle state machine transitions, and safety defenses.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
from typing import Any
import uuid

# Ensure repo root is in path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from skills.hunt_and_kill import run as run_hunt_and_kill_skill


def generate_sample_input() -> dict[str, Any]:
    """Generates baseline institutional targets representing Module 8's 5 vectors."""
    return {
        "correlation_id": f"corr-hk-{uuid.uuid4().hex[:8]}",
        "sofr_base_rate_pct": 3.88,
        "hurdle_rate_bps": 150.0,
        "targets": [
            {
                "target_id": "HK-01",
                "target_name": "Cash HY vs. CDX Basis",
                "triage_category": "Rating-Cliff Fallen Angels",
                "dislocation_vector": "Vector 2: Cash vs. Synthetic Credit Basis Compression",
                "long_leg_instrument": "Physical Senior Notes (BB Basket)",
                "long_leg_price": 92.50,
                "short_leg_instrument": "CDX NA HY 5Y Protection + Repo",
                "short_leg_price": 95.80,
                "implied_pricing": 0.958,
                "spot_pricing": 0.925,
                "cost_repo_sofr_bps": 388.0,
                "bid_ask_slippage_bps": 25.0,
                "carry_theta_div_bps": 15.0,
                "catalyst_description": "Dealer inventory rebalance & index rollover",
                "t_max_days": 15,
                "position_size_usd": 25_000_000.0,
                "adv_usd": 100_000_000.0,
                "vpin": 0.22,
                "sner": 1.1,
            },
            {
                "target_id": "HK-02",
                "target_name": "Co-Op / LMT Priming Target",
                "triage_category": "Maturity Wall & Covenant Traps",
                "dislocation_vector": "Vector 1: Capital Structure & LMT Priming Arbitrage",
                "long_leg_instrument": "1L Term Loan B (>50.1% Steering Group)",
                "long_leg_price": 68.00,
                "short_leg_instrument": "Unsecured Senior Notes / Common Equity",
                "short_leg_price": 88.00,
                "implied_pricing": 0.880,
                "spot_pricing": 0.680,
                "cost_repo_sofr_bps": 400.0,
                "bid_ask_slippage_bps": 50.0,
                "carry_theta_div_bps": 30.0,
                "catalyst_description": "8-K Item 1.01 Drop-Down / Uptier Exchange",
                "t_max_days": 45,
                "position_size_usd": 15_000_000.0,
                "adv_usd": 20_000_000.0,
                "vpin": 0.35,
                "sner": 2.4,
            },
            {
                "target_id": "HK-03",
                "target_name": "Index vs. Single-Name Vol",
                "triage_category": "Vol-Target & Risk-Parity De-Grossing",
                "dislocation_vector": "Vector 3: Dispersion & Correlation Arbitrage",
                "long_leg_instrument": "Single-Name 30D ATM Straddles",
                "long_leg_price": 14.50,
                "short_leg_instrument": "Short SPX 30D ATM Straddle (Vega-Neutral)",
                "short_leg_price": 18.20,
                "implied_pricing": 0.182,
                "spot_pricing": 0.145,
                "cost_repo_sofr_bps": 120.0,
                "bid_ask_slippage_bps": 10.0,
                "carry_theta_div_bps": 5.0,
                "catalyst_description": "Earnings dispersion realization",
                "t_max_days": 20,
                "position_size_usd": 10_000_000.0,
                "adv_usd": 150_000_000.0,
                "vpin": 0.18,
                "sner": 0.8,
            },
            {
                "target_id": "HK-05",
                "target_name": "HYG ETF Apex Ambush",
                "triage_category": "Adversarial Swarm Liquidity Traps",
                "dislocation_vector": "Vector 5: Counter-Swarm Liquidity Ambush",
                "long_leg_instrument": "HYG ETF (-185 bps NAV Discount)",
                "long_leg_price": 72.20,
                "short_leg_instrument": "Physical Basket Creation/Redemption + CDX",
                "short_leg_price": 74.05,
                "implied_pricing": 0.7405,
                "spot_pricing": 0.7220,
                "cost_repo_sofr_bps": 18.0,
                "bid_ask_slippage_bps": 22.0,
                "carry_theta_div_bps": 15.0,
                "catalyst_description": "Apex Liquidity provision at 3.2-sigma flush",
                "t_max_days": 1,
                "position_size_usd": 30_000_000.0,
                "adv_usd": 250_000_000.0,
                "vpin": 0.28,
                "sner": 1.4,
            },
        ],
        "counter_predation_inputs": [
            {
                "target_id": "HK-01",
                "short_squeeze_utilization_pct": 42.0,
                "stock_loan_borrow_cost_bps": 85.0,
                "borrow_recall_ceiling_pass": True,
                "coop_steering_position_pct": None,
                "coop_majority_verified": True,
                "overall_pass": True,
            },
            {
                "target_id": "HK-02",
                "short_squeeze_utilization_pct": 65.0,
                "stock_loan_borrow_cost_bps": 220.0,
                "borrow_recall_ceiling_pass": True,
                "coop_steering_position_pct": 54.2,
                "coop_majority_verified": True,
                "overall_pass": True,
            },
            {
                "target_id": "HK-03",
                "short_squeeze_utilization_pct": 30.0,
                "stock_loan_borrow_cost_bps": 45.0,
                "borrow_recall_ceiling_pass": True,
                "coop_steering_position_pct": None,
                "coop_majority_verified": True,
                "overall_pass": True,
            },
            {
                "target_id": "HK-05",
                "short_squeeze_utilization_pct": 20.0,
                "stock_loan_borrow_cost_bps": 30.0,
                "borrow_recall_ceiling_pass": True,
                "coop_steering_position_pct": None,
                "coop_majority_verified": True,
                "overall_pass": True,
            },
        ],
    }


def generate_prov_o_bundle(input_data: dict[str, Any], output_data: dict[str, Any]) -> dict[str, Any]:
    """Creates a W3C PROV-O audit bundle for the Hunt & Kill execution session."""
    input_hash = hashlib.sha256(json.dumps(input_data, sort_keys=True).encode("utf-8")).hexdigest()
    output_hash = hashlib.sha256(json.dumps(output_data, sort_keys=True).encode("utf-8")).hexdigest()
    bundle_id = f"prov-hk-{uuid.uuid4().hex[:12]}"

    return {
        "prov_bundle_id": bundle_id,
        "prefix": {
            "prov": "http://www.w3.org/ns/prov#",
            "adam": "https://adam.financial/ontology/v30.1#",
        },
        "entity": {
            "adam:HuntAndKillInput": {
                "prov:type": "adam:ExecutionInput",
                "adam:sha256": input_hash,
            },
            "adam:HuntAndKillOutput": {
                "prov:type": "adam:ExecutionOutput",
                "adam:sha256": output_hash,
                "adam:authorized_deployments": output_data.get("total_authorized_deployments", 0),
            },
        },
        "activity": {
            "adam:Module8Execution": {
                "prov:type": "adam:HuntAndKillOrchestration",
                "prov:startedAtTime": datetime.now(timezone.utc).isoformat(),
                "prov:endedAtTime": datetime.now(timezone.utc).isoformat(),
            }
        },
        "wasGeneratedBy": {
            "_:wgb1": {
                "prov:entity": "adam:HuntAndKillOutput",
                "prov:activity": "adam:Module8Execution",
            }
        },
        "used": {
            "_:u1": {
                "prov:activity": "adam:Module8Execution",
                "prov:entity": "adam:HuntAndKillInput",
            }
        },
    }


def render_markdown_report(output_data: dict[str, Any], prov_bundle: dict[str, Any]) -> str:
    """Renders standardized institutional Markdown report for Module 8."""
    sh = output_data.get("safety_header", {})
    header_gate = f"**{sh.get('layer_0_gate', 0)}**"
    h1_pass = 'PASS' if sh.get('h1_vpin_toxicity_pass') else 'FAIL'
    h2_pass = 'PASS' if sh.get('h2_short_utilization_pass') else 'FAIL'
    h3_pass = 'PASS' if sh.get('h3_borrow_fee_pass') else 'FAIL'
    h4_pass = 'PASS' if sh.get('h4_sacred_rights_coop_pass') else 'FAIL'
    feed_pass = 'VERIFIED' if sh.get('sha256_feed_integrity_verified') else 'UNVERIFIED'
    conf_score = f"**{sh.get('epistemic_confidence_score', 0.0)}**"
    hitl_req = 'YES' if sh.get('hitl_escalation_required') else 'NO'
    assured_score = f"**{sh.get('final_assured_score', 0.0):.2f}**"
    market_pass = 'COMPLIANT' if sh.get('market_integrity_compliance_pass') else 'NON_COMPLIANT'

    lines = [
        "# ADAM v30.1-R-Patch Module 8: Hunt & Kill Execution Report (Hardened Spec)",
        f"**Correlation ID**: `{output_data['correlation_id']}` | **Executed At**: `{output_data['timestamp']}` | **Status**: `{output_data['status']}`",
        "",
        "## 0. Layer 0 Safety & Materiality Assurance Header",
        "| Layer 0 Gate | h1 (VPIN <= 0.65) | h2 (Util <= 85%) | h3 (Borrow <= 450bps) | h4 (Co-Op Sacred Rights) | SHA-256 Feed Integrity | Epistemic Confidence | HITL Required | Final Assured Score | Market Integrity |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| {header_gate} | {h1_pass} | {h2_pass} | {h3_pass} | {h4_pass} | {feed_pass} | {conf_score} | {hitl_req} | {assured_score} | {market_pass} |",
        "",
        "## 1. Stage 1 (Locate): Forced-Flow Triage & VPIN Swarm Trap Metrics",
        "| Downgrade Prob (%) | Realized Vol 10D | VIX Anchor Ratio | Selling Pressure ($M) | LTM Int Coverage | LTM FCCR | Covenant Trap | VPIN | SNER | Swarm Trap | Position ($M) | ADV ($M) | Impact Slippage |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for idx, tm in enumerate(output_data.get("triage_metrics", [])):
        mi = output_data.get("market_impacts", [])[idx] if idx < len(output_data.get("market_impacts", [])) else {}
        cov_trap = 'ACTIVE' if tm['covenant_trap_active'] else 'INACTIVE'
        swarm_trap = 'DETECTED' if tm['swarm_trap_detected'] else 'CLEAR'
        pos_size = mi.get('position_size_dollars', 0) / 1e6
        adv_size = mi.get('average_daily_volume_dollars', 0) / 1e6
        impact_slip = mi.get('calculated_slippage_bps', 0)

        lines.append(
            f"| {tm['downgrade_probability_pct']}% | {tm['realized_vol_10d']} | {tm['vix_anchor_ratio']} | ${tm['forced_de_grossing_selling_pressure_mm']}M | "
            f"{tm['ltm_interest_coverage']}x | {tm['ltm_fccr']}x | {cov_trap} | "
            f"**{tm['vpin']:.2f}** | **{tm['sner']:.1f}x** | {swarm_trap} | "
            f"${pos_size:.1f}M | ${adv_size:.1f}M | {impact_slip} bps |"
        )

    lines.extend([
        "",
        "## 2. Structural Dislocation & Basis Ledger",
        "| Kill ID | Target / Pair | Dislocation Vector | Long Leg | Short / Hedge Leg | Gross Dislocation | Funding, Slippage & Penalty | Net Spread (Spread_Net) | VPIN / SNER | PCS Ratio | Epistemic Status | Lifecycle State | Catalyst & T_max |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---| ",
    ])

    for entry in output_data["basis_ledger"]:
        lines.append(
            f"| {entry['kill_id']} | {entry['target_pair']} | {entry['dislocation_vector']} | {entry['long_leg']} | "
            f"{entry['short_leg']} | {entry['gross_dislocation_bps']} bps | {entry['funding_slippage_cost_bps']} bps | "
            f"**{entry['net_exploitable_edge_bps']} bps** | {entry.get('vpin_sner', 'N/A')} | **{entry['pcs_score']}** | `{entry['epistemic_status']}` | `{entry['lifecycle_state']}` | {entry['convergence_catalyst_and_tmax']} |"
        )

    lines.extend([
        "",
        "## 3. Zero-Trust Order Routing & Microstructure Spec",
        "| Target ID | Air-Gapped Sanitization | Liquidity Masking Venue | Execution Algorithm | Legging Sequence |",
        "|---|---|---|---|---|",
    ])

    for spec in output_data["order_routing_specs"]:
        seq_str = "<br>".join(spec["credit_derivative_legging_sequence"])
        sanitized = 'VERIFIED' if spec.get('air_gapped_sanitization_verified') else 'UNVERIFIED'
        lines.append(f"| {spec['target_id']} | {sanitized} | {spec['liquidity_masking_venue']} | {spec['execution_algorithm']} | {seq_str} |")

    lines.extend([
        "",
        "## 4. Kill-Switch & Counter-Predation Audit Ledger",
        "| Target ID | Short Util (%) | Borrow Cost (bps) | Borrow Ceiling Pass | Co-Op Steering (%) | Co-Op Majority Verified | Overall Pass |",
        "|---|---|---|---|---|---|---|",
    ])

    for audit in output_data["counter_predation_audit"]:
        steer_str = f"{audit['coop_steering_position_pct']}%" if audit.get('coop_steering_position_pct') is not None else "N/A"
        borrow_pass = 'PASS' if audit['borrow_recall_ceiling_pass'] else 'FAIL'
        majority_ver = 'VERIFIED' if audit['coop_majority_verified'] else 'UNVERIFIED'
        overall_pass = 'PASS' if audit['overall_pass'] else 'FAIL'
        lines.append(
            f"| {audit['target_id']} | {audit['short_squeeze_utilization_pct']}% | {audit['stock_loan_borrow_cost_bps']} bps | "
            f"{borrow_pass} | {steer_str} | {majority_ver} | "
            f"**{overall_pass}** |"
        )

    lines.extend([
        "",
        "## 5. Embedded Safety Defense & Primacy Invariant Ledger",
        "| Target ID | Safety Primacy Enforced | Short Squeeze Breach | Co-Op Legality Breach | VPIN Toxicity Quarantine | Unbounded Loss Risk | Reward Override Attempted | Safety Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for sd in output_data.get("safety_defenses", []):
        lines.append(
            f"| {sd.get('target_id', 'N/A')} | {sd['safety_primacy_enforced']} | {sd['short_squeeze_ceiling_breached']} | "
            f"{sd['coop_legality_breached']} | {sd.get('vpin_toxicity_quarantine_triggered', False)} | {sd['unbounded_loss_risk_detected']} | "
            f"**{sd['reward_override_attempted']}** | **`{sd['safety_verdict']}`** |"
        )

    lines.extend([
        "",
        "## 6. Basis Convergence & Extinguish Telemetry",
        "| Target ID | Initial Spread | Current Spread | Compression (%) | 50% Scale-Out | Extinguish Triggered | Multi-Oracle 180s Consensus | Hard Kill-Switch Active | Fortress Sweep (%) |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for ext in output_data["extinguish_telemetry"]:
        oracle_ver = 'VERIFIED' if ext.get('multi_oracle_180s_consensus_verified') else 'UNVERIFIED'
        lines.append(
            f"| {ext['target_id']} | {ext['initial_spread_bps']} bps | {ext['current_spread_bps']} bps | {ext['compression_pct']}% | "
            f"{ext['scale_out_50_triggered']} | {ext['full_extinguish_triggered']} | {oracle_ver} | "
            f"{ext['hard_kill_switch_active']} | {output_data['capital_swept_to_fortress_ballast_pct']}% |"
        )

    lines.extend([
        "",
        "## 7. W3C PROV-O SHA-256 Audit Seal",
        "```json",
        json.dumps(prov_bundle, indent=2),
        "```",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="ADAM v30.1-R-Patch Module 8 Orchestrator")
    parser.add_argument("--input", help="Path to input JSON file (optional)")
    parser.add_argument("--output-dir", default="runs", help="Output directory for results")
    args = parser.parse_args()

    if args.input and os.path.exists(args.input):
        with open(args.input, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    else:
        input_data = generate_sample_input()

    # Execute skill
    output_data = run_hunt_and_kill_skill(input_data)
    prov_bundle = generate_prov_o_bundle(input_data, output_data)

    # Render report
    md_report = render_markdown_report(output_data, prov_bundle)

    os.makedirs(args.output_dir, exist_ok=True)
    report_file = os.path.join(args.output_dir, f"hunt_and_kill_execution_{output_data['correlation_id']}.md")
    json_file = os.path.join(args.output_dir, f"hunt_and_kill_execution_{output_data['correlation_id']}.json")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md_report)

    with open(json_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"input": input_data, "output": output_data, "prov_bundle": prov_bundle}, indent=2))

    print("Module 8 Hunt & Kill Orchestration Complete.")
    print(f"Report saved to: {report_file}")
    print(f"JSON data saved to: {json_file}")


if __name__ == "__main__":
    main()
