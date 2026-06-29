# Agent Schema Reference

This document details the core schemas used across the Adam Platform's autonomous agents, enforcing horizontal scalability and strict type checking.

## `AgentOutput`

The `AgentOutput` schema represents the standardized return payload from any agent (Nexus or Sentinel).

### Fields

*   `answer` (str): The final synthesized answer or analysis.
*   `sources` (List[str]): List of citations (filenames, URLs) used.
*   `confidence` (float): Conviction score ranging from 0.0 to 1.0.
*   `metadata` (Dict[str, Any]): Debug information and token usage.
*   `provenance_trace` (ProvenanceHeader): Immutable provenance trace.
*   `data` (Dict[str, Any]): The raw deterministic output payload.
*   `observed_drift` (bool): Flag indicating if logic shifted, triggering self-healing.

## `ProvenanceHeader`

The `ProvenanceHeader` ensures W3C PROV-O compliance by tracking the origin of probabilistic inferences.

### Fields

*   `git_commit_hash` (str): Environment context (PROV-O: wasGeneratedBy).
*   `timestamp` (str): ISO 8601 execution time (PROV-O: generatedAtTime).
*   `content_hash` (str): SHA-256 hash of the payload (PROV-O: value).
*   `jsonLogic_version` (str): Version of the applied jsonLogic schema.
*   `derivation_path` (str): Path of reasoning (PROV-O: wasDerivedFrom).
*   `source_data_object` (str): Source data reference (PROV-O: hadPrimarySource).
