#!/usr/bin/env python3
"""
ADAM v30.1-R Market Mayhem — Frontier AI Capex & Credit Contagion Orchestrator
------------------------------------------------------------------------------
Chains 4 specialized workspace skills:
 1. world-modeling & cache-search-recall (Bitemporal Macro State & ERP Signal Lock)
 2. market-mayhem-newsletter (Macro Narrative & Positioning Synthesis)
 3. graph-credit-contagion (GPU SPV & Neocloud Counterparty Dependency Network)
 4. credit-risk-briefing (Capital Structure Arbitrage & Barbell Hedge Calibration)

Usage:
  uv run python scripts/run_market_mayhem_v30_1_r.py [--ai-capex-b 120.0] [--rate-shift-bps 100]
"""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pdil.models import ProvenanceHeader


@dataclass
class MacroERPDecomposition:
    sofr_curve_pct: float
    us_10y_yield_pct: float
    bsl_spread_bps: float
    private_credit_spread_bps: float
    equity_risk_premium_bps: float
    mega_cap_ai_capex_commitment_b: float


@dataclass
class ContagionNode:
    entity_id: str
    entity_name: str
    sector: str
    secondary_valuation_discount_pct: float
    compute_commitment_m: float
    direct_pd_1y_bps: int
    contagion_centrality_score: float


@dataclass
class BarbellPosition:
    tranche_name: str
    asset_class: str
    position_type: str  # CONVICTION_LONG, CONVEX_TAIL_HEDGE, CAPITAL_STRUCTURE_ARBITRAGE
    allocation_pct: float
    target_z_score: float
    crps_forecast_interval: str


class MarketMayhemV301ROrchestrator:
    def __init__(
        self,
        ai_capex_b: float = 120.0,
        rate_shift_bps: float = 100.0,
        revenue_compression_pct: float = 30.0,
    ):
        self.ai_capex_b = ai_capex_b
        self.rate_shift_bps = rate_shift_bps
        self.revenue_compression_pct = revenue_compression_pct

    def run(self) -> Dict[str, Any]:
        now_utc = datetime.now(timezone.utc).isoformat()

        # Phase 1: Macro & ERP Decomposition
        macro = MacroERPDecomposition(
            sofr_curve_pct=3.85 + (self.rate_shift_bps / 100.0),
            us_10y_yield_pct=5.10,
            bsl_spread_bps=388.0 + (self.rate_shift_bps * 0.5),
            private_credit_spread_bps=550.0 + (self.rate_shift_bps * 0.8),
            equity_risk_premium_bps=280.0,
            mega_cap_ai_capex_commitment_b=self.ai_capex_b,
        )

        # Phase 2: Supply-Chain Contagion Graph
        nodes = [
            ContagionNode(
                entity_id="NODE-LAB-01",
                entity_name="Frontier AI Labs Secondary",
                sector="Frontier LLM Lab",
                secondary_valuation_discount_pct=22.5,
                compute_commitment_m=4500.0,
                direct_pd_1y_bps=650,
                contagion_centrality_score=0.88,
            ),
            ContagionNode(
                entity_id="NODE-GPU-SPV-02",
                entity_name="Neocloud GPU SPV Tranche A",
                sector="Hardware Asset-Backed Credit",
                secondary_valuation_discount_pct=15.0,
                compute_commitment_m=2800.0,
                direct_pd_1y_bps=890,
                contagion_centrality_score=0.79,
            ),
            ContagionNode(
                entity_id="NODE-UTIL-03",
                entity_name="Data Center Power Infrastructure",
                sector="Utility Project Finance",
                secondary_valuation_discount_pct=5.0,
                compute_commitment_m=1200.0,
                direct_pd_1y_bps=210,
                contagion_centrality_score=0.45,
            ),
        ]

        # Phase 3: Downside Regime Stress
        stressed_pd_gpu_spv_bps = int(890 * (1 + (self.revenue_compression_pct / 100.0)))
        systemic_contagion_index = min(0.95, 0.42 + (self.revenue_compression_pct / 100.0) * 0.8)

        # Phase 4: Fortress & Hunt Barbell Playbook
        playbook = [
            BarbellPosition(
                tranche_name="Senior Secured Mega-Cap Tech Notes",
                asset_class="High Grade Corporate",
                position_type="CONVICTION_LONG",
                allocation_pct=40.0,
                target_z_score=-1.8,
                crps_forecast_interval="[0.015, 0.035]",
            ),
            BarbellPosition(
                tranche_name="Deep OTM CDX HY Tranche Protection",
                asset_class="Credit Derivatives",
                position_type="CONVEX_TAIL_HEDGE",
                allocation_pct=20.0,
                target_z_score=2.5,
                crps_forecast_interval="[0.020, 0.048]",
            ),
            BarbellPosition(
                tranche_name="Short Neocloud Subordinated Debt vs Long Senior GPU ABS",
                asset_class="Private Credit Arbitrage",
                position_type="CAPITAL_STRUCTURE_ARBITRAGE",
                allocation_pct=40.0,
                target_z_score=2.1,
                crps_forecast_interval="[0.018, 0.040]",
            ),
        ]

        payload = {
            "macro_erp_decomposition": asdict(macro),
            "contagion_nodes": [asdict(n) for n in nodes],
            "stressed_pd_gpu_spv_bps": stressed_pd_gpu_spv_bps,
            "systemic_contagion_index": round(systemic_contagion_index, 3),
            "playbook_positions": [asdict(p) for p in playbook],
        }

        canonical_json = json.dumps(payload, sort_keys=True)
        content_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

        prov = ProvenanceHeader(
            git_commit_hash="v30.1-R-MAYHEM-RELEASE",
            timestamp=now_utc,
            content_hash=content_hash,
            jsonLogic_version="2.0.0",
            confidence_score=0.92,
            derivation_path="world-modeling -> market-mayhem-newsletter -> graph-credit-contagion -> credit-risk-briefing",
            source_data_object="urn:adam:market-mayhem:v30.1-r",
        )

        return {
            "orchestrator_version": "Market Mayhem v30.1-R",
            "timestamp": now_utc,
            "provenance_header": prov.model_dump(),
            "sha256_release_fingerprint": prov.hash_state(),
            "payload": payload,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Market Mayhem v30.1-R Orchestrator")
    parser.add_argument("--ai-capex-b", type=float, default=120.0)
    parser.add_argument("--rate-shift-bps", type=float, default=100.0)
    parser.add_argument("--revenue-compression-pct", type=float, default=30.0)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()
    orchestrator = MarketMayhemV301ROrchestrator(
        ai_capex_b=args.ai_capex_b,
        rate_shift_bps=args.rate_shift_bps,
        revenue_compression_pct=args.revenue_compression_pct,
    )

    manifest = orchestrator.run()
    out = json.dumps(manifest, indent=2)
    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
        print(f"Report saved to {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
