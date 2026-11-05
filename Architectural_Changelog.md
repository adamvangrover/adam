# Architectural Changelog

## 1. Architectural Changes
*   **Horizontal Scalability Improvements:**
    *   Mandated strict type checking across agents by mapping references to `AgentOutput` in `src/schemas/core_types.py`.
    *   Unified interfaces for decoupled model execution and independent dependency resolution.
*   **Vertical Scalability Improvements:**
    *   Strengthened the integration layer between the probabilistic engine (Adam) and the deterministic actions (PDIL middleware).
    *   Reinforced W3C PROV-O compliant metadata validation within the `SecurityGovernanceGatekeeper` to satisfy "Context-First" documentation requirements and ensure AI context windows can trace data sources.
    *   Retained redundant `compute_deterministic_hash` implementations in `src/schemas/core_types.py` as an explicit dual-check fallback mechanism, satisfying architectural requirements for maximum assurance.
*   **Self-Healing & Testing Robustness:**
    *   `DriftIntelligenceLayer` continuously monitors and corrects data representation divergences.
    *   Introduced property-based stress tests (`hypothesis`) in `tests/test_provenance.py` focusing on W3C PROV-O source data object grounding (`check_grounding`).
    *   Added static typing (`ruff` compliant) to `tests/test_pdil_property.py` to ensure deterministic codebase standards.

## 2. Upgraded PDIL Component
*   The logic inside `src/schemas/core_types.py` was cleaned of general duplication to reflect a solid deterministic foundation before bridging to the Probabilistic Engine, explicitly excluding the redundant hash check which serves as an added invariant guard.
*   The Probabilistic-to-Deterministic Integration Layer (PDIL) mapping was established and documented in `docs/architecture/PDIL_MAPPING.md`.

## 3. Provenance Test Suite
*   Property-based testing is now implemented robustly in `tests/test_pdil_property.py` to validate PROV-O integrity over multiple payload edge-cases. Type hints enforce strict adherence.

## 4. Observed Drift
*   **Observed Drift:** No logic drift observed from the core functionality of the schemas. The previous implementation featured duplicate imports which were removed, but the redundant `compute_deterministic_hash` function was deliberately preserved for high-assurance invariant checking.
