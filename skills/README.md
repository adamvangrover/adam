# ADAM Governed Agent Skills Registry

This directory contains deterministic, governed agent skills operating within the ADAM neuro-symbolic framework.

## Registered Skills

### 1. Epistemic State & Grounding Validator (`skills/epistemic_grounding.py`)
- **Manifest**: `schemas/agent_skill_manifest.json` (skill_id: `epistemic_grounding`)
- **Input Schema**: `schemas/epistemic_grounding_input.json`
- **Output Schema**: `schemas/epistemic_grounding_output.json`
- **Purpose**: Verifies agent assertions against actual repository artifacts and commits. Computes `promotion_allowed` deterministically via policy rather than model selection.

### 2. Quantitative Risk Engine Interoperability (`skills/pd_model_challenger.py`)
- **Manifest**: (skill_id: `pd_model_challenger`)
- **Input Schema**: `schemas/pd_model_challenger_input.json`
- **Output Schema**: `schemas/pd_model_challenger_output.json`
- **Purpose**: Calculates $EL = PD \times LGD \times EAD$ for credit proposals using fractional units ($0 \le PD \le 1$), compares challenger vs incumbent under specified tolerance thresholds, and emits non-authoritative proposals.

## Governance & Authority Invariants
- Skills run with `authority_level: proposal_only`. No skill can directly grant canonical state admission or approve credit decisions.
- Inputs and outputs are validated against JSON Schemas before and after process execution.
- All executions generate traceable provenance records and idempotency hashes.
