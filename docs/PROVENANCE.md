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

### 2a. Synthetic scenario path (publication backfill)

`data/scenarios/2026H2_path.json` is a **synthetic** daily path (business days 2026-04-01 → 2026-11-24)
for S&P 500, Nasdaq 100, UST 2Y/10Y, VIX, HY OAS, Brent, gold, DXY and Bitcoin. It is not market data.

- Built by `scripts/publications/build_scenario_path.py` from `config/scenarios/2026H2_path.yaml`
  (seeded `random.Random(seed)` Brownian bridge; rerunning reproduces the file byte-for-byte).
- Pinned to the levels quoted in the committed `newsletters/market_mayhem_2026*.md` narratives (e.g.
  S&P 500 7,592.40 and UST 10Y 5.102% on 2026-11-12), so backfilled editions agree with them. Those
  narratives are themselves unverified; their anchors are not treated as real prices.
- The file and every record derived from it carry `_estimated: true`; rendered pages show
  "Illustrative scenario — not investment advice". Editions never attribute quotes to real people and
  never describe events that are not in the path.
- Price targets (`showcase/data/publications/price_targets_latest.json`) are computed by
  `core/engine/valuation_utils.py` from the illustrative inputs in `config/valuation/coverage_2026.yaml`
  (also `_estimated: true`) with the path's 10Y as the risk-free rate.

## 3. Ledgers

- Immutable decision events: `core/governance/` (hash-chained ledger).
- Bitemporal schema for storage backends: `data/schemas/unified_ledger_bitemporal.sql`.
- PROV-O corroboration across disjoint sources: `core/engine/prov_o_consensus.py`.
