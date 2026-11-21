# Provenance Convention

One page. Applies to agent outputs, config values and data files.

## 1. Agent outputs — `provenance_trace`

`AgentOutput.provenance_trace: List[ProvenanceHeader]` (`core/schemas/agent_schema.py`).

| Field (`src/pdil/models.py::ProvenanceHeader`) | PROV-O | Meaning |
|---|---|---|
| `source_data_object` | `hadPrimarySource` | URI/path of the data the output came from. Required for grounding. |
| `derivation_path` | `wasDerivedFrom` | How the answer was reached (e.g. `TemplateAgentV30.analyze`). |
| `content_hash` | `value` | SHA-256 of the canonical JSON payload. |
| `jsonLogic_version` | — | Version of the rule set applied. |
| `git_commit_hash`, `timestamp` | `wasGeneratedBy`, `generatedAtTime` | Code version and time. |
| `confidence_score` | — | Conviction, 0.0–1.0. |

Rules:

- Empty list (the default) means **ungrounded**. `AgentOutput.check_grounding()` is True only when the list
  is non-empty and every header has a `source_data_object`.
- Build headers with `src/pdil/models.py::build_provenance_header(data, ...)`; it computes `content_hash`.
- A material decision (rating, sizing, capital, trade) with an empty trace must not run autonomously.
- One header per source. A single header or dict passed in is wrapped into a list.

## 2. Values — `source` or `_estimated: true`

Every hand-entered value in `config/` or `data/` is either sourced or flagged:

```yaml
AAPL:
  sector: Technology
  rating: AA+
  source: "S&P issuer rating, 2026-06"   # where the values come from

HYG:
  sector: Credit
  rating: BB
  _estimated: true                       # hand-picked proxy, not sourced data
```

- Code that fabricates a default at runtime sets `_estimated: True` on the produced dict.
- Missing data is skipped and logged, not filled with an unflagged default
  (e.g. `MetaOrchestrator._positions_from_allocation` lists them in `_skipped_symbols`).
- This is a convention, not a lint rule. Reviewers check it.

## 3. Ledgers

- Immutable decision events: `core/governance/` (hash-chained ledger).
- Bitemporal schema for storage backends: `data/schemas/unified_ledger_bitemporal.sql`.
- PROV-O corroboration across disjoint sources: `core/engine/prov_o_consensus.py`.
