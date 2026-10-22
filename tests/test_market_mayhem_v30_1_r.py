"""
Pytest Verification Suite for Market Mayhem v30.1-R Frontier AI Capex & Credit Contagion Orchestrator.
"""

import pytest
from scripts.run_market_mayhem_v30_1_r import MarketMayhemV301ROrchestrator


def test_market_mayhem_v30_1_r_end_to_end():
    orchestrator = MarketMayhemV301ROrchestrator(
        ai_capex_b=120.0,
        rate_shift_bps=100.0,
        revenue_compression_pct=30.0,
    )

    manifest = orchestrator.run()

    assert manifest["orchestrator_version"] == "Market Mayhem v30.1-R"
    assert len(manifest["sha256_release_fingerprint"]) == 64

    payload = manifest["payload"]
    macro = payload["macro_erp_decomposition"]
    assert macro["sofr_curve_pct"] == 4.85
    assert macro["mega_cap_ai_capex_commitment_b"] == 120.0

    nodes = payload["contagion_nodes"]
    assert len(nodes) == 3
    assert nodes[0]["entity_id"] == "NODE-LAB-01"

    assert payload["stressed_pd_gpu_spv_bps"] == 1157
    assert payload["systemic_contagion_index"] > 0.50

    playbook = payload["playbook_positions"]
    assert len(playbook) == 3
    assert playbook[0]["position_type"] == "CONVICTION_LONG"
    assert playbook[1]["position_type"] == "CONVEX_TAIL_HEDGE"
    assert playbook[2]["position_type"] == "CAPITAL_STRUCTURE_ARBITRAGE"
