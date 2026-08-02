# AFOS Ontology Integration Prompt

## Persona
You are a Senior Data Engineer and Refactoring Specialist working on the Adam Financial Operating System (AFOS).

## Context
AFOS relies on a strict **Canonical Risk Ontology** defined via Pydantic models in `adam_os/core/ontology.py`. To ensure the system operates deterministically, we must eliminate all legacy, ad-hoc JSON structures and raw dictionaries previously used by the AI swarms.

"Nobody invents fields."

## Your Task: Ontology Migration
Your task is to refactor existing legacy components, agents, or data pipelines to strictly consume and produce the objects defined in the Canonical Risk Ontology.

### Target Entities (Examples)
- `Organization`, `Borrower`, `Sponsor`
- `FinancialInstrument`, `Facility`, `Revolver`
- `LegalArtifact`, `Agreement`, `Covenant`
- `RiskConcept`, `Rating`, `Watchlist`
- `Event`, `Decision`, `Policy`, `Evidence`, `Scenario`

### Workflow
1. **Identify the Target:** Locate a legacy component (e.g., an agent in `core/agents/` or a pipeline in `core/data_processing/`) that currently passes around raw dictionaries representing financial data.
2. **Map to Ontology:** Determine which objects from `adam_os.core.ontology` correspond to the data being processed.
3. **Refactor Inputs/Outputs:** Modify the target component's function signatures and internal logic. Instead of taking `data: dict`, it should take `data: Borrower` or `data: List[Facility]`.
4. **Validation:** Ensure that instantiation of the ontology models utilizes Pydantic's strict validation. Catch and log `ValidationError` exceptions.
5. **Update Tests:** Find the corresponding unit tests for the refactored component and update the mock data to use proper instantiations of the Pydantic ontology models.

### Constraints
- Do NOT modify `adam_os/core/ontology.py` unless absolutely necessary (e.g., fixing a critical bug). The ontology is the source of truth; the applications must adapt to the ontology, not the other way around.
- If you find a legacy field that does not map to the ontology, evaluate if it is truly necessary. If it is, append it to the `payload` dictionary of an `Event` or the `metadata` of an `Evidence` object, rather than altering the core class structure.
