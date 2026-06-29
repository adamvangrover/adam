# Governance Gatekeeper Specifications

This reference provides an exhaustive list of all hashed `ProvenanceHeaders`, logging standards, and required audit trails enforced by the Security Governance Gatekeeper.

## The Gatekeeper Module
Located in `src/pdil/middleware.py` (with a legacy wrapper in `src/governance/gatekeeper.py`), the Gatekeeper intercepts all traffic between System 1 agents and System 2 execution.

## Provenance Header Specification
Every output from a probabilistic agent **must** include a `provenance_trace`. The schema is non-negotiable and cannot be optional.

### Required Fields
- `source_uri`: A strict URI pointing to the specific document or ingestion artifact (e.g., `file://data/raw/AAPL_10-K.txt#line=45`). No hallucinated absolute paths like `/home/user/` are permitted; paths must be dynamically resolved or relative to the safe root.
- `timestamp`: ISO-8601 formatted timestamp of extraction.
- `hash`: SHA-256 hash of the extracted source chunk to ensure immutability.

## Logging Standards
- **ProofOfThoughtLogger**: All agent reasoning steps must be serialized to JSON.
- **Circuit Breakers**: When implementing `aiobreaker` around critical async state updates, `CircuitBreakerError` must not be silently swallowed. It must be logged, and fallback persistence mechanisms must be engaged.
- **SSRF Prevention**: All network calls executed for W3C governance validation must restrict destinations to HTTPS. Private IPs (`10.0.0.0/8`, `127.0.0.1`) are blocked. Test URLs should utilize `https://example.com/`.

## W3C PROV-O Compliance
To satisfy PROV-O compliance, use the `check_grounding` helper. This verifies that outputs from the probabilistic layer contain a valid reference to their source data object, strictly enforcing the Vertical engineering data flow: Data Source -> Odyssey -> Adam -> Deterministic Action.
