---
name: headline-arena-hitl
description: Human-in-the-loop forecasting loop for Headline Arena (headlinearena.com) — stage, review, dispatch, settle, learn. Use when reviewing performance, staging or submitting predictions to open challenges, auditing settlements, or tuning priors/confidence for the ADAM-Macro-Sentinel agent.
---

# Headline Arena HITL Loop

Workspace: repo root. Package: `headline_arena/`. Memory: `data/memory/`. Creds: `.ha_credentials.json`.

## Loop (run in order; each step is a deterministic tool)

| # | Step | Command | Output |
|---|------|---------|--------|
| 1 | Learn from last round | `python3 -m headline_arena.learn data/memory/round<N-1>_live_dispatch_results.json` | `settlement_ledger.jsonl`, `calibration.json` |
| 2 | Update inputs | Edit `ASSET_UNIVERSE` / priors in `analytics.py` using `calibration.json` (`suggested_conf_cap`, `suggested_neutral_prior`) | diff for operator |
| 3 | Stage (no network writes) | `python3 -m headline_arena.dispatch --all-open --dry-run` | `hitl_review_dossier.md`, `hitl_staged_forecasts.json` |
| 4 | **HARD STOP** | Present summary table; wait for operator `DISPATCH_CONFIRMED` | — |
| 5 | Dispatch verbatim | `python3 -m headline_arena.submit_staged --confirm --out round<N>_live_dispatch_results.json` | submission log with `prediction_id`s |
| 6 | After `resolve_at` (21:00 UTC) | back to step 1 with round N | — |

Never use `dispatch --dispatch-confirmed` for live sends: it regenerates forecasts. `submit_staged` posts exactly what was reviewed.

## Verification rules (HITL integrity)
- Every number in an operator-facing artifact (hash, weight, price, rank) must be **copied from tool output or the staged JSON**, never typed from memory. If not read, say "see staged JSON".
- Quote leaderboard rank/accuracy from `GET /api/v1/eval/leaderboard?limit=100` (match `agent_id`), not from prior notes.
- Disclose that `order_flow_imbalance`, `edgar_triage_signal`, catalysts in `analytics.py` are **hand-set narrative parameters**, not live feeds.
- Do not apply calibration suggestions silently; show the diff and let the operator approve.

## API gotchas
- Always send `Accept-Encoding: gzip, deflate` (Cloudflare zstd breaks `requests.json()`).
- `/api/v1/eval/challenges/active` → `{"challenges":[{"challenge":{...}}]}`; unwrap `item.get("challenge", item)`.
- Challenge detail: `GET /api/v1/eval/challenges/{id}` → `open_price`, `close_price`, `result`, `dead_zone_pct`, crowd counts.
- Submit: `POST /api/v1/eval/challenges/{id}/predict` (Bearer token from `dispatch.get_fresh_token`).
- Retry GETs only. Never auto-retry POST (duplicate/revision risk); log timeout as UNKNOWN and check before resending.
- Agent profile endpoints (`/agents/{id}`) 404 — use the leaderboard.

## Settlement facts (verified Oct 2026)
- Result = % change from session open to first resolver run ≥ `resolve_at`; exactly ±DZ resolves neutral.
- Dead zones are **per asset**: ZN 0.05%, DXY 0.15%, VIX 0.80% (open_price often null), others 0.30%. Always read `dead_zone_pct` from the challenge.
- VIX resolved neutral R3–R5. GC/ZS frequently trapped in DZ.
- Energy collapse → equity/copper relief rally (R5: CL −2.7%, RB −7.6% → ES +0.6%, HG +0.9%).

## Calibration heuristics
- Confidence is scored: cap ≈ shrunk hit rate + 0.10 (see `learn.py`), floor 0.50. Avoid blanket 0.72.
- Templated rationales score ~69 on quality; asset-specific, evidence-cited text is the next upgrade.
- Monte Carlo seed is `zlib.crc32(challenge_id)` — staged output is reproducible.
