"""
ADAM OS v30.1 / AFOS V3 — Harness2 Multi-Agent Benchmark CLI Runner
Compliance: Federal Reserve SR 11-7, OCC 2026-13, Basel CRE 31
"""

from __future__ import annotations

import argparse
import json
import sys

from adam_finance.risk_adapter_gate import (
    CreditSymbolicMetrics,
    EpistemicStatus,
    TimesFMRiskSurface,
    evaluate_neuro_symbolic_decision,
)
from afos_v3.prov_o_consensus import (
    AgentAssertion,
    ProvEntity,
    ProvOGraphConsensusValidator,
    SourceModality,
)


def run_cloudwave_lbo_scenario(
    obligor: str,
    enable_prov_o: bool = True,
    enforce_cre31_floors: bool = True,
) -> dict[str, object]:
    """Executes the CloudWave Systems Inc. distressed TMT LBO stress test."""
    prov_summary: dict[str, object] = {}

    if enable_prov_o:
        prov_validator = ProvOGraphConsensusValidator()
        prov_validator.register_entity(
            ProvEntity(
                uri="urn:sec:edgar:0001844921-26-000412:10-Q:Note14",
                modality=SourceModality.SEC_EDGAR,
                content_hash_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                is_terminal_root=True,
            )
        )
        prov_validator.register_entity(
            ProvEntity(
                uri="urn:syndicate:loantape:cloudwave_tlb:q3_2026",
                modality=SourceModality.LOAN_TAPE_SQL,
                content_hash_sha256="ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
                is_terminal_root=True,
            )
        )
        prov_validator.register_entity(
            ProvEntity(
                uri="urn:market:timesfm:tail_risk_surface:tmt_software_b",
                modality=SourceModality.TIMESFM_QUANT,
                content_hash_sha256="4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce",
                is_terminal_root=True,
            )
        )
        prov_validator.register_entity(
            ProvEntity(
                uri="urn:legal:credit_agreement:cloudwave_2024_amended_inc",
                modality=SourceModality.LEGAL_INDENTURE,
                content_hash_sha256="8c3666d92634d0b1357c92b2fc6368d37449b948",
                is_terminal_root=True,
            )
        )

        assertions = [
            AgentAssertion(
                agent_id="eve_v2_forensic_underwriter",
                claim_id="claim_01_covenant_ebitda_reconciliation",
                assertion_text="True Covenant EBITDA is $174.8M vs $215M reported; FCCR is 1.12x.",
                derived_entity_uris={
                    "urn:sec:edgar:0001844921-26-000412:10-Q:Note14",
                    "urn:syndicate:loantape:cloudwave_tlb:q3_2026",
                },
                epistemic_tag=EpistemicStatus.CALCULATED.value,
            ),
            AgentAssertion(
                agent_id="timesfm_market_risk_agent",
                claim_id="claim_02_tail_risk_collapse",
                assertion_text="TimesFM conditional 60d tail quantile q_01 drops to 61.5 par (VaR_99 = 27.7%).",
                derived_entity_uris={"urn:market:timesfm:tail_risk_surface:tmt_software_b"},
                epistemic_tag=EpistemicStatus.MODEL_EST.value,
            ),
            AgentAssertion(
                agent_id="module_8_hunt_and_kill",
                claim_id="claim_03_lmt_priming_trap",
                assertion_text="Negative covenants lack J.Crew IP blocker; $160M collateral leakage capacity.",
                derived_entity_uris={"urn:legal:credit_agreement:cloudwave_2024_amended_inc"},
                epistemic_tag=EpistemicStatus.VERIFIED.value,
            ),
        ]

        consensus_passed, prov_summary = prov_validator.evaluate_triangulated_consensus(assertions)
        if not consensus_passed:
            raise RuntimeError(f"PROV-O consensus failed: {prov_summary}")

    market_risk_surface = TimesFMRiskSurface(
        asset_id="CLOUDWAVE_1400M_TLB",
        horizon_days=60,
        adapter_param_count=2460000,
        quantiles={
            0.01: -0.195,
            0.05: -0.118,
            0.10: -0.065,
            0.50: 0.004,
            0.90: 0.042,
            0.95: 0.081,
            0.99: 0.125,
        },
        trailing_252d_exceedances_99=4,
        epistemic_tag=EpistemicStatus.MODEL_EST,
    )

    root_sources = (
        set(prov_summary.get("unique_terminal_roots", []))
        if enable_prov_o
        else set()
    )

    cohort_floor = 0.092 if enforce_cre31_floors else 0.0005

    credit_metrics = CreditSymbolicMetrics(
        obligor_id=obligor,
        fccr_reported=1.28,
        fccr_true_covenant=1.12,
        gross_leverage=8.38,
        dscr_stressed_p05=0.91,
        pd_pit=0.084,
        pd_ttc=0.038,
        cohort_75th_pct_pd_floor=cohort_floor,
        brier_score_oot=0.041,
        lmt_uncapped_leakage_pct=0.245,
        accounting_identity_residual_usd_m=0.000,
        prov_o_root_sources=root_sources,
    )

    decision_result = evaluate_neuro_symbolic_decision(
        credit=credit_metrics,
        market=market_risk_surface,
        s_grounding=4.9,
        s_downside=4.8,
        s_reusability=4.85,
        materiality_tier=2,
        human_approved=False,
    )

    return {
        "scenario": "tmt_lbo_distressed",
        "obligor_id": obligor,
        "prov_o_summary": prov_summary,
        "decision_result": decision_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harness2 Multi-Agent Benchmark Runner (ADAM OS v30.1)"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="tmt_lbo_distressed",
        help="Benchmark scenario name",
    )
    parser.add_argument(
        "--obligor",
        type=str,
        default="CLOUDWAVE_SYSTEMS_LBO",
        help="Obligor identifier",
    )
    parser.add_argument(
        "--enable-prov-o",
        action="store_true",
        default=True,
        help="Enable W3C PROV-O root-source disjointness validation",
    )
    parser.add_argument(
        "--enforce-cre31-floors",
        action="store_true",
        default=True,
        help="Enforce Basel III Endgame CRE 31 A-IRB cohort floors",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output raw JSON only",
    )

    args = parser.parse_args()

    if args.scenario != "tmt_lbo_distressed":
        print(f"Unknown scenario: {args.scenario}", file=sys.stderr)
        sys.exit(1)

    result = run_cloudwave_lbo_scenario(
        obligor=args.obligor,
        enable_prov_o=args.enable_prov_o,
        enforce_cre31_floors=args.enforce_cre31_floors,
    )

    decision = result["decision_result"]
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
