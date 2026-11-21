# SNAPSHOT: afos_v3

**Purpose:** AFOS V3 zero-trust ingestion kernel — PROV-O disjoint-root consensus, STS/mTLS gateway, schema bounds, triangulated 3-feed consensus with 75/25 Fortress/Hunt lockdown.

**Status:** Fully mined. All code was promoted; nothing remains here but this note.

| Original | Promoted to | Why |
|---|---|---|
| `prov_o_consensus.py` | `core/engine/prov_o_consensus.py` | Unique: corroboration must trace to disjoint authoritative roots. Complements (does not replace) `core/engine/consensus_engine.py` (weighted vote arbitration) and `consensus_engine_v2.py` (memory-matrix house view). |
| `gateway/zero_trust_gateway.py` | `core/zero_trust/gateway.py` | Dependency of the consensus engine; used by `scripts/sre_eval_harness.py`. |
| `validation/schema_bounds.py` | `core/zero_trust/schema_bounds.py` | Dependency of the consensus engine. |
| `consensus/triangulated_engine.py` | `core/zero_trust/triangulated_consensus.py` | Kept over `src/ingestion/zero_trust/consensus_engine.py`: async, gateway-integrated, lockdown circuit breaker, covered by `tests/test_zero_trust_remediation.py`. |
