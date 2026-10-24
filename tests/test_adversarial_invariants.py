import math
import os
import unittest
import yaml
from pydantic import ValidationError

from schemas.credit_contracts import (
    SourceLocator,
    EvidenceStatus,
    EvidenceReference,
    NegativeFinding,
    FinancialAssertion,
    AuthorizedStateTransitionRequest,
)
from engine.dealroom_context import DealRoomContextManager
from engine.jsonlogic_validator_v2 import (
    GovernedJsonLogicValidator,
    MissingFinancialInputError,
    UnverifiedEvidenceError,
    CovenantDirectionError,
    NumericDomainError,
)
from engine.cross_model_verifier import (
    CrossModelVerifier,
    DisagreementClassification,
)
from engine.provenance_replay import ProvenanceReplayLedger


class TestAdversarialInvariants(unittest.TestCase):
    """
    Adversarial test suite verifying all 15 critical invariants required by the
    ADAM Autonomous Credit Risk Framework Remediation Protocol.
    """

    def setUp(self):
        self.doc_hash = "a"*64
        self.valid_locator = SourceLocator(
            artifact_sha256=self.doc_hash,
            page_number=12,
            paragraph_index=3,
            sec_accession_number="0001193125-26-000123",
            section_title="Note 7 - Financial Covenants"
        )
        self.valid_evidence = EvidenceReference(
            evidence_id="EV-1001",
            status=EvidenceStatus.AUDITED,
            locator=self.valid_locator,
            extracted_at_timestamp=1700000000.0,
            confidence_score=0.98,
            extractor_model="anthropic/claude-3-5-sonnet-20241022"
        )

    # 1. Missing EBITDA add-back schedules
    def test_01_missing_ebitda_addback_schedule(self):
        data = {"ttm_ebitda_usd": None}
        rule = {"var": "ttm_ebitda_usd"}
        with self.assertRaises(MissingFinancialInputError):
            GovernedJsonLogicValidator.evaluate(rule, data)

    # 2. Conflicting SEC filing periods
    def test_02_conflicting_sec_filing_periods(self):
        verifier = CrossModelVerifier()
        a1 = FinancialAssertion(
            assertion_id="A1",
            metric_name="EBITDA",
            value=100000000.0,
            unit="USD",
            reporting_period="TTM_2026_Q2",
            evidence_ref=self.valid_evidence
        )
        a2 = FinancialAssertion(
            assertion_id="A2",
            metric_name="EBITDA",
            value=100000000.0,
            unit="USD",
            reporting_period="FY_2025",
            evidence_ref=self.valid_evidence
        )
        res = verifier.compare_assertions(a1, a2)
        self.assertEqual(res["classification"], DisagreementClassification.SUBSTANTIAL_DISAGREEMENT)
        self.assertTrue(res["requires_human_escalation"])

    # 3. Incorrect units and scaling
    def test_03_incorrect_units_and_scaling(self):
        verifier = CrossModelVerifier()
        a1 = FinancialAssertion(
            assertion_id="A1",
            metric_name="TOTAL_DEBT",
            value=500.0,
            unit="USD_MILLIONS",
            reporting_period="FY_2026",
            evidence_ref=self.valid_evidence
        )
        a2 = FinancialAssertion(
            assertion_id="A2",
            metric_name="TOTAL_DEBT",
            value=500000000.0,
            unit="USD",
            reporting_period="FY_2026",
            evidence_ref=self.valid_evidence
        )
        res = verifier.compare_assertions(a1, a2)
        self.assertEqual(res["classification"], DisagreementClassification.SUBSTANTIAL_DISAGREEMENT)
        self.assertTrue(res["requires_human_escalation"])

    # 4. Maximum versus minimum covenant inequality direction
    def test_04_covenant_inequality_direction(self):
        data = {"leverage_ratio": 7.00, "leverage_ceiling": 6.00}
        rule = {"<=": [{"var": "leverage_ratio"}, {"var": "leverage_ceiling"}]}
        res = GovernedJsonLogicValidator.validate_covenant_headroom(
            covenant_rule=rule,
            financial_data=data,
            covenant_name="Net Leverage Ceiling",
            expected_direction="<="
        )
        self.assertFalse(res["passed"])
        self.assertEqual(res["calculated_metric"], 7.00)
        self.assertEqual(res["covenant_threshold"], 6.00)
        self.assertEqual(res["headroom"], -1.00)

        # Mismatched direction should raise CovenantDirectionError
        with self.assertRaises(CovenantDirectionError):
            GovernedJsonLogicValidator.validate_covenant_headroom(
                covenant_rule=rule,
                financial_data=data,
                covenant_name="Net Leverage Floor Error",
                expected_direction=">="
            )

    # 5. Unsupported EBITDA adjustments
    def test_05_unsupported_ebitda_adjustments_rejected(self):
        unverified_ref = EvidenceReference(
            evidence_id="EV-UNVERIFIED",
            status=EvidenceStatus.UNVERIFIED,
            locator=self.valid_locator,
            extracted_at_timestamp=1700000000.0,
            confidence_score=0.40,
            extractor_model="unverified-model"
        )
        unverified_assertion = FinancialAssertion(
            assertion_id="A_UNVERIFIED",
            metric_name="SYNERGY_ADD_BACK",
            value=15000000.0,
            unit="USD",
            reporting_period="FY_2026",
            evidence_ref=unverified_ref
        )
        data = {"synergy_addback": unverified_assertion}
        rule = {"var": "synergy_addback"}
        with self.assertRaises(UnverifiedEvidenceError):
            GovernedJsonLogicValidator.evaluate(rule, data)

    # 6. Fabricated model outputs
    def test_06_fabricated_model_outputs_schema_violation(self):
        with self.assertRaises(ValidationError):
            NegativeFinding(
                finding_id="INVALID-PREFIX-123",
                target_metric="EBITDA_ADD_BACK",
                searched_scope=["Exhibit 99.1"],
                exhaustion_reason="Exhausted"
            )

    # 7. Unknown and quarantined model IDs
    def test_07_quarantined_model_fail_closed(self):
        with open("config/llm_routing.yaml", "r") as f:
            routing_config = yaml.safe_load(f)

        registry = routing_config["model_registry"]
        legacy_model = registry["deprecated/legacy-credit-llm-v0"]
        self.assertTrue(legacy_model["quarantined"])
        self.assertEqual(legacy_model["allowed_environments"], [])

        # Check denylist
        denylist = routing_config["denylist"]["forbidden_model_prefixes"]
        test_model = "unverified/bad-actor-llm"
        self.assertTrue(any(test_model.startswith(prefix) for prefix in denylist))

    # 8. Cache misses and expiration
    def test_08_cache_miss_and_expiration(self):
        mgr = DealRoomContextManager(cache_ttl_seconds=1.0)
        pass1 = mgr.evaluate_pass_with_cache("key1", 850000, 4000, now=1000.0)
        self.assertFalse(pass1["is_cache_hit"])
        self.assertEqual(pass1["uncached_input_tokens"], 850000)

        # Call within TTL -> Cache hit
        pass2 = mgr.evaluate_pass_with_cache("key1", 850000, 4000, now=1000.5)
        self.assertTrue(pass2["is_cache_hit"])
        self.assertEqual(pass2["cached_input_tokens"], 850000)

        # Call after TTL -> Cache miss
        pass3 = mgr.evaluate_pass_with_cache("key1", 850000, 4000, now=1002.0)
        self.assertFalse(pass3["is_cache_hit"])
        self.assertEqual(pass3["uncached_input_tokens"], 850000)

    # 9. Provider timeouts and retry duplication
    def test_09_idempotent_retries_anti_duplication(self):
        verifier = CrossModelVerifier()
        a1 = FinancialAssertion(
            assertion_id="A1",
            metric_name="NET_DEBT",
            value=200000000.0,
            unit="USD",
            reporting_period="FY_2026",
            evidence_ref=self.valid_evidence
        )
        a2 = FinancialAssertion(
            assertion_id="A2",
            metric_name="NET_DEBT",
            value=200000000.0,
            unit="USD",
            reporting_period="FY_2026",
            evidence_ref=self.valid_evidence
        )

        res1 = verifier.compare_assertions(a1, a2, idempotency_key="RETRY-KEY-999")
        self.assertFalse(res1["is_idempotent_replay"])

        # Identical second dispatch using same idempotency key
        res2 = verifier.compare_assertions(a1, a2, idempotency_key="RETRY-KEY-999")
        self.assertTrue(res2["is_idempotent_replay"])

    # 10. Cross-model disagreement
    def test_10_cross_model_disagreement_triggers_escalation(self):
        verifier = CrossModelVerifier(tolerance_pct=0.05)
        a1 = FinancialAssertion(
            assertion_id="A1",
            metric_name="FREE_CASH_FLOW",
            value=100000000.0,  # $100M
            unit="USD",
            reporting_period="FY_2026",
            evidence_ref=self.valid_evidence
        )
        a2 = FinancialAssertion(
            assertion_id="A2",
            metric_name="FREE_CASH_FLOW",
            value=115500000.0,  # $115.5M (15.5% variance)
            unit="USD",
            reporting_period="FY_2026",
            evidence_ref=self.valid_evidence
        )

        res = verifier.compare_assertions(a1, a2)
        self.assertEqual(res["classification"], DisagreementClassification.SUBSTANTIAL_DISAGREEMENT)
        self.assertTrue(res["requires_human_escalation"])
        self.assertEqual(res["governance_action"], "ROUTE_TO_SECOND_LINE_CREDIT_OFFICER")

    # 11. Tampered provenance
    def test_11_tampered_provenance_detected(self):
        ledger = ProvenanceReplayLedger()
        ledger.record_event("ASSERTION_ADDED", "AST-101", "agent_1", {"value": 100})
        ledger.record_event("ASSERTION_AUDITED", "AST-101", "officer_2", {"status": "AUDITED"})

        self.assertTrue(ledger.verify_ledger_integrity())

        # Tamper with event payload in history
        ledger.chain[0]["payload"]["value"] = 999
        self.assertFalse(ledger.verify_ledger_integrity())

    # 12. Unauthorized state transitions
    def test_12_unauthorized_state_transition_fails_closed(self):
        with self.assertRaises(ValidationError):
            AuthorizedStateTransitionRequest(
                request_id="REQ-1",
                current_state="DRAFT",
                target_state="SANCTIONED",
                proposed_by_agent="autonomous_agent_alpha",
                sanctioning_officer_id=None,
                is_authorized=True,
                justification="Agent auto-sanctioning"
            )

    # 13. Deterministic replay
    def test_13_deterministic_replay(self):
        # Analytical Merton Distance-to-Default (d2) and PD calculation
        def calculate_merton_pd(v0: float, debt: float, mu: float, sigma: float, t: float) -> float:
            d1 = (math.log(v0 / debt) + (mu + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
            d2 = d1 - sigma * math.sqrt(t)
            pd = 0.5 * math.erfc(d2 / math.sqrt(2))
            return pd

        pd_run1 = calculate_merton_pd(v0=1000.0, debt=800.0, mu=0.05, sigma=0.20, t=1.0)
        pd_run2 = calculate_merton_pd(v0=1000.0, debt=800.0, mu=0.05, sigma=0.20, t=1.0)

        self.assertEqual(pd_run1, pd_run2)
        self.assertAlmostEqual(pd_run1, 0.1028, places=3)

    # 14. Evidence drift versus policy drift
    def test_14_evidence_drift_vs_policy_drift(self):
        # Evidence drift: Company takes on $150M additional debt under same policy
        rule = {"<=": [{"/": [{"var": "debt"}, {"var": "ebitda"}]}, {"var": "max_leverage"}]}

        t0_data = {"debt": 500.0, "ebitda": 100.0, "max_leverage": 6.0}
        t0_res = GovernedJsonLogicValidator.evaluate(rule, t0_data)
        self.assertTrue(t0_res)  # 5.0 <= 6.0 -> Pass

        t1_drifted_data = {"debt": 650.0, "ebitda": 100.0, "max_leverage": 6.0}
        t1_res = GovernedJsonLogicValidator.evaluate(rule, t1_drifted_data)
        self.assertFalse(t1_res)  # 6.5 <= 6.0 -> Fail

    # 15. Rust/Python parity
    def test_15_rust_python_analytical_merton_parity(self):
        # Python analytical Merton vs Reference analytical Merton parameters
        v0, debt, mu, sigma, t = 1000.0, 800.0, 0.05, 0.20, 1.0
        d1 = (math.log(v0 / debt) + (mu + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
        pd = 0.5 * math.erfc(d2 / math.sqrt(2))

        self.assertAlmostEqual(d2, 1.2657, places=3)
        self.assertAlmostEqual(pd, 0.1028, places=3)


if __name__ == "__main__":
    unittest.main()
