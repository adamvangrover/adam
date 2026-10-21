import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple


def compute_sha256(data: Any) -> str:
    if isinstance(data, dict) or isinstance(data, list):
        payload = json.dumps(data, sort_keys=True)
    else:
        payload = str(data)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


DOMAINS = [
    "credit_facilities",
    "quantitative_models",
    "software_repositories",
    "project_backlogs",
    "research_intelligence"
]

TIERS = ["hot_memory", "warm_searchable", "cold_audit_storage"]


def process_bitemporal_lineage(artifacts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    lineage_records = []
    now_iso = datetime.now(timezone.utc).isoformat()

    by_domain = {d: 0 for d in DOMAINS}
    by_tier = {t: 0 for t in TIERS}

    for art in artifacts:
        art_id = art.get("artifact_id", "unknown-artifact")
        domain = art.get("domain", "credit_facilities")
        tier = art.get("current_lifecycle_tier", "hot_memory")

        if domain in by_domain:
            by_domain[domain] += 1
        if tier in by_tier:
            by_tier[tier] += 1

        payload = art.get("content_payload", art.get("attributes", {}))
        sha_hash = compute_sha256(payload)

        valid_at = art.get("valid_at", now_iso)
        transaction_at = art.get("transaction_at", now_iso)

        prov_o_ref = f"prov:entity/{art_id}#hash={sha_hash[:8]}"

        lineage_records.append({
            "artifact_id": art_id,
            "sha256_hash": sha_hash,
            "valid_at": valid_at,
            "transaction_at": transaction_at,
            "prov_o_ref": prov_o_ref
        })

    summary = {
        "total_artifacts": len(artifacts),
        "artifacts_by_domain": by_domain,
        "artifacts_by_tier": by_tier
    }

    return lineage_records, summary


def process_dependency_mapping(edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    graph_edges = []
    critical_paths = []
    cross_domain_edges = 0

    for edge in edges:
        src = edge.get("source_id", "")
        tgt = edge.get("target_id", "")
        rel = edge.get("relation_type", "DEPENDS_ON")
        src_dom = edge.get("source_domain")
        tgt_dom = edge.get("target_domain")

        edge_obj = {
            "source_id": src,
            "target_id": tgt,
            "relation_type": rel
        }
        if src_dom:
            edge_obj["source_domain"] = src_dom
        if tgt_dom:
            edge_obj["target_domain"] = tgt_dom

        graph_edges.append(edge_obj)

        if src_dom and tgt_dom and src_dom != tgt_dom:
            cross_domain_edges += 1

        if "REGIME" in rel or "IMPACTS" in rel or "REMEDIATES" in rel or "CRITICAL" in rel or rel == "MACRO_SHIFT_AFFECTS_FACILITY":
            critical_paths.append(f"{src} --[{rel}]--> {tgt}")

    return {
        "edge_count": len(graph_edges),
        "cross_domain_edges_count": cross_domain_edges,
        "edges": graph_edges,
        "critical_paths": critical_paths
    }


def evaluate_lifecycle_transitions(artifacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    transitions = []

    for art in artifacts:
        art_id = art.get("artifact_id", "")
        curr_tier = art.get("current_lifecycle_tier", "hot_memory")
        status = str(art.get("status", "")).upper()

        new_tier = curr_tier
        reason = "NO_TRANSITION"

        # Archival rules
        if status in ["CLOSED", "RELEASED", "MERGED", "RESOLVED"]:
            if curr_tier == "hot_memory":
                new_tier = "warm_searchable"
                reason = f"Artifact status '{status}' triggered hot_memory -> warm_searchable transition."
        elif status in ["AUDITED", "REGULATORY_ARCHIVED", "DEPRECATED"]:
            if curr_tier in ["hot_memory", "warm_searchable"]:
                new_tier = "cold_audit_storage"
                reason = f"Artifact status '{status}' triggered transition to cold_audit_storage."

        if new_tier != curr_tier:
            signal = f"INVALIDATE_CACHE:{art_id}:{curr_tier}->{new_tier}"
            transitions.append({
                "artifact_id": art_id,
                "previous_tier": curr_tier,
                "new_tier": new_tier,
                "transition_reason": reason,
                "cache_invalidation_signal": signal
            })

    return transitions


def execute_point_in_time_replay(artifacts: List[Dict[str, Any]], query: Dict[str, Any]) -> Dict[str, Any]:
    as_of_valid = query.get("as_of_valid_at", "9999-12-31T23:59:59")
    as_of_tx = query.get("as_of_transaction_at", "9999-12-31T23:59:59")
    target_domains = query.get("target_domains", [])

    active_artifacts = []

    for art in artifacts:
        valid_at = art.get("valid_at", "")
        tx_at = art.get("transaction_at", "")
        domain = art.get("domain", "")

        domain_match = not target_domains or domain in target_domains
        bitemporal_match = (valid_at <= as_of_valid) and (tx_at <= as_of_tx)

        if domain_match and bitemporal_match:
            active_artifacts.append(art.get("artifact_id", ""))

    return {
        "as_of_valid_at": as_of_valid,
        "as_of_transaction_at": as_of_tx,
        "active_artifacts": active_artifacts
    }


def run(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Skill 2 — librarian-archivist
    Maintains strongly-typed unified catalog, bitemporal lineage, dependency mapping,
    and tiered archival lifecycle transitions.
    """
    correlation_id = input_data.get("correlation_id", "unknown-correlation-id")
    operation = input_data.get("operation", "catalog_register")
    artifacts = input_data.get("artifacts", [])
    edges = input_data.get("dependency_edges", [])
    replay_query = input_data.get("replay_query")

    reason_codes = []

    # 1. Bitemporal Lineage & Fingerprinting
    lineage_records, catalog_summary = process_bitemporal_lineage(artifacts)
    reason_codes.append(f"CATALOGED_{catalog_summary['total_artifacts']}_ARTIFACTS")

    # 2. Dependency Mapping
    dependency_graph = process_dependency_mapping(edges)
    if dependency_graph["edge_count"] > 0:
        reason_codes.append(f"MAPPED_{dependency_graph['edge_count']}_DEPENDENCY_EDGES")

    # 3. Lifecycle Transitions
    transitions = evaluate_lifecycle_transitions(artifacts)
    if transitions:
        reason_codes.append(f"EXECUTED_{len(transitions)}_LIFECYCLE_TRANSITIONS")

    # 4. Point-In-Time Replay if requested
    replay_snapshot = None
    if operation == "point_in_time_replay" or replay_query:
        query = replay_query or {}
        replay_snapshot = execute_point_in_time_replay(artifacts, query)
        reason_codes.append(f"POINT_IN_TIME_REPLAY_ACTIVE_COUNT_{len(replay_snapshot['active_artifacts'])}")

    status = "PASSED" if artifacts else "COMPLETED"

    prov_o_audit_trail = [
        {
            "activity": "librarian_archivist_indexing",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id,
            "operation": operation,
            "total_indexed": catalog_summary["total_artifacts"],
            "input_hash": compute_sha256(input_data)
        }
    ]

    return {
        "skill_id": "librarian_archivist",
        "status": status,
        "correlation_id": correlation_id,
        "catalog_summary": catalog_summary,
        "bitemporal_lineage_records": lineage_records,
        "dependency_graph": dependency_graph,
        "lifecycle_transitions": transitions,
        "replay_snapshot": replay_snapshot,
        "reason_codes": reason_codes,
        "prov_o_audit_trail": prov_o_audit_trail
    }
