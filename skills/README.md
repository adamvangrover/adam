# ADAM Governed Agent Skills Registry

This directory contains deterministic, governed agent skills operating within the ADAM neuro-symbolic framework.

## Registered Skills

### 1. Epistemic State & Grounding Validator (`skills/epistemic_grounding.py`)
- **Manifest**: `schemas/agent_skill_manifest.json` (skill_id: `epistemic_grounding`)
- **Input Schema**: `schemas/epistemic_grounding_input.json`
- **Output Schema**: `schemas/epistemic_grounding_output.json`
- **Purpose**: Verifies agent assertions against actual repository artifacts and commits. Computes `promotion_allowed` deterministically via policy rather than model selection.

### 2. Quantitative Risk Engine Interoperability (`skills/pd_model_challenger.py`)
- **Manifest**: `schemas/pd_model_challenger_manifest.json` (skill_id: `pd_model_challenger`)
- **Input Schema**: `schemas/pd_model_challenger_input.json`
- **Output Schema**: `schemas/pd_model_challenger_output.json`
- **Purpose**: Calculates $EL = PD \times LGD \times EAD$ for credit proposals using fractional units ($0 \le PD \le 1$), compares challenger vs incumbent under specified tolerance thresholds, and emits non-authoritative proposals.

### 3. LLM-as-Judge & Arbiter (`skills/llm_as_judge_arbiter.py`)
- **Manifest**: `schemas/llm_as_judge_arbiter_manifest.json` (skill_id: `llm_as_judge_arbiter`)
- **Input Schema**: `schemas/llm_as_judge_arbiter_input.json`
- **Output Schema**: `schemas/llm_as_judge_arbiter_output.json`
- **Capabilities**: `multi_criteria_evaluation`, `pairwise_adversarial_adjudication`, `neuro_symbolic_tripwire_gating`, `human_alignment_calibration`
- **Purpose**: Performs multi-criteria scoring rubrics across empirical groundedness, reasoning rigor & downside completeness, and policy compliance (SR 11-7, SNC classifications, liquidity runway limits); head-to-head pairwise debate adjudication with position & verbosity bias mitigation and token density tracking; neuro-symbolic boolean tripwire gating (e.g., DSCR < 1.05x, unauthorized covenant add-backs, liquidity runway breach); and human alignment calibration computing Cohen's Kappa ($\kappa$) and Brier scores against Investment and Credit Committee outcomes.

### 4. Librarian & Archivist (`skills/librarian_archivist.py`)
- **Manifest**: `schemas/librarian_archivist_manifest.json` (skill_id: `librarian_archivist`)
- **Input Schema**: `schemas/librarian_archivist_input.json`
- **Output Schema**: `schemas/librarian_archivist_output.json`
- **Capabilities**: `unified_asset_ontology`, `bitemporal_lineage_replay`, `cross_entity_dependency_mapping`, `tiered_archival_lifecycle`
- **Purpose**: Maintains a strongly typed catalog connecting 5 primary institutional domains (Credit Facilities, Quantitative Models, Software Repositories, Project Backlogs, Research Intelligence) with bitemporal lineage (`valid_at`, `transaction_at`), SHA-256 fingerprinting, PROV-O metadata, cross-entity dependency mapping with critical path extraction, point-in-time replay, and tiered archival lifecycle management (`hot_memory` -> `warm_searchable` -> `cold_audit_storage`) with automated cache invalidation signals.

## Governance & Authority Invariants
- Skills run with `authority_level: proposal_only`. No skill can directly grant canonical state admission or approve credit decisions.
- Inputs and outputs are validated against JSON Schemas before and after process execution.
- All executions generate traceable provenance records and idempotency hashes.
