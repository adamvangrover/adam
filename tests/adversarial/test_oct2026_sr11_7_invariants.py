"""
Adversarial Machine Verification Suite for October 2026 Model Frontier Upgrade.
Enforces SR 11-7 model risk management invariants, token economics, and W3C PROV-O negative findings.
"""

import unittest
import sys
import os
from pydantic import ValidationError

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.config.models import ModelSpec, RouterConfig, SR11_7_BLOCKED_MODEL_ERROR
from core.agents.argon_system2_client import ArgonSystem2Client
from core.symbolic.covenant_qmc_bridge import (
    CovenantQMCBridge,
    SR117UnverifiedTelemetryError,
    JsonLogicEvaluator
)
from core.telemetry.prov_o import UnverifiedNegativeFinding


class TestOctober2026SR117Invariants(unittest.TestCase):

    def test_sr11_7_quarantine_blocks_gpt_6_1_astra(self):
        with self.assertRaises((ValidationError, SR11_7_BLOCKED_MODEL_ERROR)) as ctx:
            ModelSpec(provider="openai", model_id="gpt-6.1-astra", role="autonomous_execution")
        self.assertIn("SR 11-7 VIOLATION", str(ctx.exception))

        with self.assertRaises((ValidationError, SR11_7_BLOCKED_MODEL_ERROR)) as ctx:
            ModelSpec(provider="openai", model_id="GPT-6.1-ASTRA-UNALIGNED", role="autonomous_execution")
        self.assertIn("SR 11-7 VIOLATION", str(ctx.exception))

        with self.assertRaises(SR11_7_BLOCKED_MODEL_ERROR) as ctx:
            ArgonSystem2Client(model_id="gpt-6.1-astra")
        self.assertIn("SR 11-7 VIOLATION", str(ctx.exception))

    def test_argon_context_cache_deal_room_economics(self):
        client = ArgonSystem2Client(model_id="gemini-4-argon")
        economics = client.calculate_deal_room_economics(
            token_count=850_000,
            total_passes=12,
            output_tokens_per_pass=4_000
        )
        self.assertEqual(economics.base_write_cost_usd, 1.7000)
        self.assertEqual(economics.cached_read_cost_usd, 0.9350)
        self.assertEqual(economics.output_cost_usd, 0.4800)
        self.assertEqual(economics.total_cost_usd, 3.1150)
        self.assertEqual(economics.uncached_total_cost_usd, 20.8800)
        self.assertEqual(economics.competitor_total_cost_usd, 51.7200)
        self.assertEqual(economics.savings_vs_uncached_pct, 85.08)
        self.assertEqual(economics.savings_vs_competitor_pct, 93.98)

    def test_sr11_7_negative_findings_schema_and_execution_halt(self):
        client = ArgonSystem2Client(model_id="gemini-4-argon")
        bridge = CovenantQMCBridge(seed=42)

        extraction_result = client.execute_specialist_pass(
            deal_id="DEAL-ROOM-BSL-001",
            agent_role="ebitda_add_back_verifier",
            task_prompt="Reconcile restructuring footnote to Article I definition",
            context_override={
                "simulate_omitted_footnote": True,
                "target_metric": "ebitda_add_back_restructuring_cap",
                "document": "Form 10-K FY2026",
                "accession_number": "0000320193-26-000102"
            }
        )

        self.assertEqual(extraction_result["status"], "UNVERIFIED_NEGATIVE_FINDING")
        self.assertTrue(extraction_result["pause_execution"])
        self.assertEqual(extraction_result["negative_finding"]["token"], UnverifiedNegativeFinding.TOKEN)

        with self.assertRaises(SR117UnverifiedTelemetryError) as ctx:
            bridge.inspect_and_validate_telemetry(extraction_result)

        self.assertIn("SR 11-7 HALT", str(ctx.exception))
        self.assertIn("ebitda_add_back_restructuring_cap", str(ctx.exception))

    def test_deterministic_jsonlogic_and_sobol_qmc_simulation(self):
        bridge = CovenantQMCBridge(seed=42)
        verified_telemetry = {
            "consolidated_ebitda": 450_000_000.0,
            "total_funded_debt": 1_800_000_000.0,
            "senior_secured_debt": 1_125_000_000.0,
            "fixed_charges": 110_000_000.0,
            "max_total_leverage_covenant": 4.50,
            "max_secured_leverage_covenant": 3.00,
            "min_fixed_charge_coverage_covenant": 1.25
        }

        validated = bridge.inspect_and_validate_telemetry(verified_telemetry)
        covenant_eval = bridge.evaluate_covenants(validated)

        self.assertTrue(covenant_eval["all_covenants_passed"])
        self.assertEqual(covenant_eval["metrics"]["actual_total_leverage"], 4.0)
        self.assertEqual(covenant_eval["metrics"]["actual_secured_leverage"], 2.5)
        self.assertEqual(covenant_eval["metrics"]["headroom_total_leverage"], 0.5)

        qmc_results = bridge.run_qmc_stress_simulation(
            validated,
            n_samples_power_of_two=12,
            ebitda_volatility=0.20
        )

        self.assertEqual(qmc_results["sample_count"], 4096)
        self.assertEqual(qmc_results["simulation_status"], "DETERMINISTIC_QMC_CONVERGED")
        self.assertGreater(qmc_results["probability_of_covenant_breach"], 0.0)
        self.assertLess(qmc_results["probability_of_covenant_breach"], 1.0)
        self.assertGreater(qmc_results["p99_stressed_leverage"], 4.50)

    def test_router_configuration_loading(self):
        config = RouterConfig.load_from_yaml(os.path.join(REPO_ROOT, "config/llm_routing.yaml"))
        self.assertEqual(config.tier_1_primary_system2.model_id, "gemini-4-argon")
        self.assertEqual(config.tier_1_primary_system2.provider, "google")
        self.assertEqual(config.tier_1_primary_system2.pricing.cached_read_per_million, 0.10)
        self.assertEqual(config.deal_room_cache.default_ttl_seconds, 7200)


if __name__ == "__main__":
    unittest.main()
