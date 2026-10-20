import os
import pytest
from skills.librarian_archivist import run as run_librarian_archivist


def test_librarian_archivist_5_domain_catalog_registration():
    input_data = {
        "correlation_id": "test-lib-001",
        "operation": "catalog_register",
        "artifacts": [
            {
                "artifact_id": "FAC-2026-TMT-001",
                "domain": "credit_facilities",
                "entity_type": "credit_facility",
                "attributes": {"cusip": "123456789", "facility_size": 250000000},
                "current_lifecycle_tier": "hot_memory",
                "status": "ACTIVE"
            },
            {
                "artifact_id": "MODEL-MERTON-V2",
                "domain": "quantitative_models",
                "entity_type": "merton_solver",
                "attributes": {"version": "2.1.0", "solver_type": "scrambled_sobol"},
                "current_lifecycle_tier": "warm_searchable",
                "status": "RELEASED"
            },
            {
                "artifact_id": "GIT-COMMIT-A1B2C3D",
                "domain": "software_repositories",
                "entity_type": "git_commit",
                "attributes": {"commit_sha": "a1b2c3d4e5f6", "repo": "adam-kernel"},
                "current_lifecycle_tier": "hot_memory",
                "status": "MERGED"
            },
            {
                "artifact_id": "TICKET-RISK-2026-99",
                "domain": "project_backlogs",
                "entity_type": "risk_review_ticket",
                "attributes": {"priority": "P0", "sprint": "2026-Q4-S1"},
                "current_lifecycle_tier": "hot_memory",
                "status": "CLOSED"
            },
            {
                "artifact_id": "RESEARCH-MAYHEM-ISSUE-14",
                "domain": "research_intelligence",
                "entity_type": "macro_pulse",
                "attributes": {"title": "Credit Crunch & Liquidity Volatility"},
                "current_lifecycle_tier": "warm_searchable",
                "status": "PUBLISHED"
            }
        ],
        "dependency_edges": [
            {
                "source_id": "RESEARCH-MAYHEM-ISSUE-14",
                "target_id": "FAC-2026-TMT-001",
                "relation_type": "IMPACTS_FACILITY"
            },
            {
                "source_id": "MODEL-MERTON-V2",
                "target_id": "FAC-2026-TMT-001",
                "relation_type": "EVALUATES_PROBABILITY_OF_DEFAULT"
            }
        ]
    }

    result = run_librarian_archivist(input_data)
    assert result["skill_id"] == "librarian_archivist"
    assert result["status"] == "PASSED"
    summary = result["catalog_summary"]
    assert summary["total_artifacts"] == 5

    # Verify all 5 domains present in catalog
    by_domain = summary["artifacts_by_domain"]
    assert by_domain["credit_facilities"] == 1
    assert by_domain["quantitative_models"] == 1
    assert by_domain["software_repositories"] == 1
    assert by_domain["project_backlogs"] == 1
    assert by_domain["research_intelligence"] == 1

    # Verify bitemporal lineage and SHA-256 fingerprinting
    lineage = result["bitemporal_lineage_records"]
    assert len(lineage) == 5
    for rec in lineage:
        assert len(rec["sha256_hash"]) == 64
        assert rec["prov_o_ref"].startswith("prov:entity/")

    # Verify cross-entity dependency mapping
    dep = result["dependency_graph"]
    assert dep["edge_count"] == 2
    assert len(dep["critical_paths"]) >= 1

    # Verify lifecycle transition
    transitions = result["lifecycle_transitions"]
    assert len(transitions) == 2  # GIT-COMMIT (MERGED) and TICKET (CLOSED) transition hot -> warm
    for t in transitions:
        assert t["cache_invalidation_signal"].startswith("INVALIDATE_CACHE:")


def test_librarian_archivist_point_in_time_replay():
    input_data = {
        "correlation_id": "test-lib-002",
        "operation": "point_in_time_replay",
        "artifacts": [
            {
                "artifact_id": "FAC-HIST-2025",
                "domain": "credit_facilities",
                "entity_type": "facility",
                "valid_at": "2025-06-01T00:00:00Z",
                "transaction_at": "2025-06-01T00:00:00Z"
            },
            {
                "artifact_id": "FAC-FUTURE-2027",
                "domain": "credit_facilities",
                "entity_type": "facility",
                "valid_at": "2027-01-01T00:00:00Z",
                "transaction_at": "2027-01-01T00:00:00Z"
            }
        ],
        "replay_query": {
            "as_of_valid_at": "2026-01-01T00:00:00Z",
            "as_of_transaction_at": "2026-01-01T00:00:00Z"
        }
    }

    result = run_librarian_archivist(input_data)
    replay = result["replay_snapshot"]
    assert replay is not None
    assert "FAC-HIST-2025" in replay["active_artifacts"]
    assert "FAC-FUTURE-2027" not in replay["active_artifacts"]
