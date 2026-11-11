"""
ADAM OS v30.1 / AFOS V3 — W3C PROV-O Lineage & Disjointness Consensus Engine
Module: afos_v3/prov_o_consensus.py
Compliance: Federal Reserve SR 11-7, OCC 2026-13 (Agent Lineage Auditability)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SourceModality(str, Enum):
    SEC_EDGAR = "SEC_EDGAR"              # Primary regulatory filing (10-K, 10-Q, 8-K)
    LOAN_TAPE_SQL = "LOAN_TAPE_SQL"      # Core banking / servicing ledger record
    TIMESFM_QUANT = "TIMESFM_QUANT"      # Patched time-series quantile surface
    MARKET_TELEMETRY = "MARKET_TELEMETRY"  # Real-time SOFR / CDS / bond runs
    LEGAL_INDENTURE = "LEGAL_INDENTURE"  # Executed credit agreement / indenture PDF
    SYNTHETIC_AGENT = "SYNTHETIC_AGENT"  # Intermediate agent inference (non-terminal)


@dataclass(frozen=True)
class ProvEntity:
    """Represents a node in the W3C PROV-O graph."""
    uri: str
    modality: SourceModality
    content_hash_sha256: str
    is_terminal_root: bool = False
    derived_from: set[str] = field(default_factory=set)


@dataclass
class AgentAssertion:
    """An analytical finding emitted by an agent during consensus voting."""
    agent_id: str
    claim_id: str
    assertion_text: str
    derived_entity_uris: set[str]
    epistemic_tag: str  # [VERIFIED], [CALCULATED], [MODEL_EST]


class ProvOGraphConsensusValidator:
    """
    Traverses PROV-O derivation paths to verify that corroborating agent claims
    originate from mutually disjoint, authoritative root entities.
    """

    def __init__(self) -> None:
        self._entities: dict[str, ProvEntity] = {}

    def register_entity(self, entity: ProvEntity) -> None:
        self._entities[entity.uri] = entity

    def get_terminal_root_sources(self, entity_uri: str, visited: set[str] | None = None) -> set[str]:
        """Recursively resolves an entity to its primary external root sources."""
        if visited is None:
            visited = set()

        if entity_uri in visited:
            # Cycle detected in derivation graph
            return set()
        visited.add(entity_uri)

        entity = self._entities.get(entity_uri)
        if not entity:
            return set()

        if entity.is_terminal_root:
            return {entity.uri}

        roots = set()
        for parent_uri in entity.derived_from:
            roots |= self.get_terminal_root_sources(parent_uri, visited.copy())
        return roots

    def evaluate_triangulated_consensus(
        self,
        assertions: list[AgentAssertion],
        min_disjoint_roots: int = 3,
        min_distinct_modalities: int = 3,
        max_source_jaccard_overlap: float = 0.20,
    ) -> tuple[bool, dict[str, object]]:
        """
        Validates whether a pool of agent assertions achieves genuine neuro-symbolic consensus.

        Rules:
        1. Root Disjointness: Primary root source sets must not exceed max Jaccard overlap.
        2. Modality Diversity: Must encompass at least min_distinct_modalities (e.g. Legal != Quant != SQL).
        3. Strict Provenance Integrity: Every intermediate entity must be registered and non-empty.
        """
        if len(assertions) < min_disjoint_roots:
            return False, {
                "verdict": "FAIL_INSUFFICIENT_ASSERTIONS",
                "reason": f"Required {min_disjoint_roots} assertions, got {len(assertions)}",
            }

        agent_root_map: dict[str, set[str]] = {}
        agent_modalities: dict[str, set[SourceModality]] = {}
        all_unique_roots: set[str] = set()

        for assertion in assertions:
            roots_for_agent: set[str] = set()
            modalities_for_agent: set[SourceModality] = set()

            for uri in assertion.derived_entity_uris:
                terminal_roots = self.get_terminal_root_sources(uri)
                roots_for_agent |= terminal_roots

                for root_uri in terminal_roots:
                    root_entity = self._entities.get(root_uri)
                    if root_entity:
                        modalities_for_agent.add(root_entity.modality)

            agent_root_map[assertion.agent_id] = roots_for_agent
            agent_modalities[assertion.agent_id] = modalities_for_agent
            all_unique_roots |= roots_for_agent

        # Check Pairwise Jaccard Overlap
        agent_ids = list(agent_root_map.keys())
        max_overlap_observed = 0.0
        colluding_pair = None

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                id_a, id_b = agent_ids[i], agent_ids[j]
                roots_a, roots_b = agent_root_map[id_a], agent_root_map[id_b]

                if not roots_a or not roots_b:
                    return False, {
                        "verdict": "FAIL_UNGROUNDED_AGENT",
                        "reason": f"Agent {id_a if not roots_a else id_b} has no terminal PROV-O roots.",
                    }

                intersection = len(roots_a & roots_b)
                union = len(roots_a | roots_b)
                jaccard = intersection / union if union > 0 else 0.0

                if jaccard > max_overlap_observed:
                    max_overlap_observed = jaccard
                    colluding_pair = (id_a, id_b)

        # Aggregate Modality Coverage
        aggregated_modalities = set.union(*agent_modalities.values()) if agent_modalities else set()

        # Final Verification Criteria
        is_disjoint = max_overlap_observed <= max_source_jaccard_overlap
        has_sufficient_roots = len(all_unique_roots) >= min_disjoint_roots
        has_sufficient_modalities = len(aggregated_modalities) >= min_distinct_modalities

        passed = is_disjoint and has_sufficient_roots and has_sufficient_modalities

        audit_payload = {
            "verdict": "CONSENSUS_PROVENANCE_PASS" if passed else "CONSENSUS_PROVENANCE_FAIL",
            "passed": passed,
            "unique_terminal_roots_count": len(all_unique_roots),
            "unique_terminal_roots": list(all_unique_roots),
            "max_pairwise_jaccard_overlap": round(max_overlap_observed, 3),
            "colluding_pair": colluding_pair if not is_disjoint else None,
            "distinct_modalities_count": len(aggregated_modalities),
            "modalities_present": [m.value for m in aggregated_modalities],
            "g_provenance_flag": 1 if passed else 0,
        }

        return passed, audit_payload
