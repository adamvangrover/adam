"""
Unit test suite for W3C PROV-O Root-Source Disjointness Engine (afos_v3/prov_o_consensus.py).
Compliance: Federal Reserve SR 11-7, OCC 2026-13 (Agent Lineage Auditability).
"""

from __future__ import annotations

import pytest

from afos_v3.prov_o_consensus import (
    AgentAssertion,
    ProvEntity,
    ProvOGraphConsensusValidator,
    SourceModality,
)


@pytest.fixture
def populated_validator() -> ProvOGraphConsensusValidator:
    validator = ProvOGraphConsensusValidator()
    # Register Terminal Primary Roots
    validator.register_entity(
        ProvEntity(
            uri="urn:sec:edgar:0001844921-26-000412:10-Q:Note14",
            modality=SourceModality.SEC_EDGAR,
            content_hash_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            is_terminal_root=True,
        )
    )
    validator.register_entity(
        ProvEntity(
            uri="urn:syndicate:loantape:cloudwave_tlb:q3_2026",
            modality=SourceModality.LOAN_TAPE_SQL,
            content_hash_sha256="ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
            is_terminal_root=True,
        )
    )
    validator.register_entity(
        ProvEntity(
            uri="urn:market:timesfm:tail_risk_surface:tmt_software_b",
            modality=SourceModality.TIMESFM_QUANT,
            content_hash_sha256="4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce",
            is_terminal_root=True,
        )
    )
    validator.register_entity(
        ProvEntity(
            uri="urn:legal:credit_agreement:cloudwave_2024_amended_inc",
            modality=SourceModality.LEGAL_INDENTURE,
            content_hash_sha256="8c3666d92634d0b1357c92b2fc6368d37449b948",
            is_terminal_root=True,
        )
    )
    return validator


def test_cloudwave_disjointness_consensus(
    populated_validator: ProvOGraphConsensusValidator,
) -> None:
    validator = populated_validator

    assertions = [
        AgentAssertion(
            agent_id="eve_v2_forensic_underwriter",
            claim_id="claim_01_covenant_ebitda_reconciliation",
            assertion_text="True Covenant EBITDA is $174.8M vs $215M reported; FCCR is 1.12x.",
            derived_entity_uris={
                "urn:sec:edgar:0001844921-26-000412:10-Q:Note14",
                "urn:syndicate:loantape:cloudwave_tlb:q3_2026",
            },
            epistemic_tag="[CALCULATED]",
        ),
        AgentAssertion(
            agent_id="timesfm_market_risk_agent",
            claim_id="claim_02_tail_risk_collapse",
            assertion_text="TimesFM conditional 60d tail quantile q_01 drops to 61.5 par (VaR_99 = 27.7%).",
            derived_entity_uris={"urn:market:timesfm:tail_risk_surface:tmt_software_b"},
            epistemic_tag="[MODEL_EST]",
        ),
        AgentAssertion(
            agent_id="module_8_hunt_and_kill",
            claim_id="claim_03_lmt_priming_trap",
            assertion_text="Negative covenants lack J.Crew IP blocker; $160M collateral leakage capacity.",
            derived_entity_uris={"urn:legal:credit_agreement:cloudwave_2024_amended_inc"},
            epistemic_tag="[VERIFIED]",
        ),
    ]

    consensus_passed, audit_payload = validator.evaluate_triangulated_consensus(assertions)

    assert consensus_passed is True
    assert audit_payload["verdict"] == "CONSENSUS_PROVENANCE_PASS"
    assert audit_payload["unique_terminal_roots_count"] == 4
    assert audit_payload["distinct_modalities_count"] == 4
    assert audit_payload["max_pairwise_jaccard_overlap"] == 0.0
    assert audit_payload["g_provenance_flag"] == 1


def test_derivation_chain_and_cycle_handling(
    populated_validator: ProvOGraphConsensusValidator,
) -> None:
    validator = populated_validator

    # Create intermediate entity derived from SEC filing
    validator.register_entity(
        ProvEntity(
            uri="urn:agent:memo:preliminary_ebitda_audit",
            modality=SourceModality.SYNTHETIC_AGENT,
            content_hash_sha256="1111111111111111111111111111111111111111111111111111111111111111",
            derived_from={"urn:sec:edgar:0001844921-26-000412:10-Q:Note14"},
        )
    )

    roots = validator.get_terminal_root_sources("urn:agent:memo:preliminary_ebitda_audit")
    assert roots == {"urn:sec:edgar:0001844921-26-000412:10-Q:Note14"}

    # Introduce a cyclic derivation
    validator.register_entity(
        ProvEntity(
            uri="urn:cycle:node_a",
            modality=SourceModality.SYNTHETIC_AGENT,
            content_hash_sha256="aaaa",
            derived_from={"urn:cycle:node_b"},
        )
    )
    validator.register_entity(
        ProvEntity(
            uri="urn:cycle:node_b",
            modality=SourceModality.SYNTHETIC_AGENT,
            content_hash_sha256="bbbb",
            derived_from={"urn:cycle:node_a"},
        )
    )
    # Must handle cycle gracefully and return empty set
    cyclic_roots = validator.get_terminal_root_sources("urn:cycle:node_a")
    assert cyclic_roots == set()


def test_synthetic_consensus_rejection_high_jaccard(
    populated_validator: ProvOGraphConsensusValidator,
) -> None:
    validator = populated_validator

    # Agent A and Agent B rely on the exact same root source (collusion / echo chamber)
    assertions = [
        AgentAssertion(
            agent_id="agent_alpha",
            claim_id="claim_a",
            assertion_text="Finding A",
            derived_entity_uris={"urn:sec:edgar:0001844921-26-000412:10-Q:Note14"},
            epistemic_tag="[VERIFIED]",
        ),
        AgentAssertion(
            agent_id="agent_beta",
            claim_id="claim_b",
            assertion_text="Finding B (echoing finding A)",
            derived_entity_uris={"urn:sec:edgar:0001844921-26-000412:10-Q:Note14"},
            epistemic_tag="[VERIFIED]",
        ),
        AgentAssertion(
            agent_id="agent_gamma",
            claim_id="claim_c",
            assertion_text="Finding C",
            derived_entity_uris={"urn:syndicate:loantape:cloudwave_tlb:q3_2026"},
            epistemic_tag="[VERIFIED]",
        ),
    ]

    consensus_passed, audit_payload = validator.evaluate_triangulated_consensus(assertions)
    assert consensus_passed is False
    assert audit_payload["verdict"] == "CONSENSUS_PROVENANCE_FAIL"
    assert audit_payload["max_pairwise_jaccard_overlap"] == 1.0
    assert audit_payload["colluding_pair"] == ("agent_alpha", "agent_beta")
    assert audit_payload["g_provenance_flag"] == 0


def test_insufficient_assertions_and_ungrounded_agent(
    populated_validator: ProvOGraphConsensusValidator,
) -> None:
    validator = populated_validator

    # Less than min_disjoint_roots (default 3)
    passed, payload = validator.evaluate_triangulated_consensus([])
    assert passed is False
    assert payload["verdict"] == "FAIL_INSUFFICIENT_ASSERTIONS"

    # Ungrounded agent assertion
    assertions = [
        AgentAssertion(
            agent_id="agent_1",
            claim_id="c1",
            assertion_text="t1",
            derived_entity_uris={"urn:sec:edgar:0001844921-26-000412:10-Q:Note14"},
            epistemic_tag="[VERIFIED]",
        ),
        AgentAssertion(
            agent_id="agent_2",
            claim_id="c2",
            assertion_text="t2",
            derived_entity_uris={"urn:syndicate:loantape:cloudwave_tlb:q3_2026"},
            epistemic_tag="[VERIFIED]",
        ),
        AgentAssertion(
            agent_id="agent_unregistered",
            claim_id="c3",
            assertion_text="t3",
            derived_entity_uris={"urn:nonexistent:source"},
            epistemic_tag="[MODEL_EST]",
        ),
    ]
    passed, payload = validator.evaluate_triangulated_consensus(assertions)
    assert passed is False
    assert payload["verdict"] == "FAIL_UNGROUNDED_AGENT"
