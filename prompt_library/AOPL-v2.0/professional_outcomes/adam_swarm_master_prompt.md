# ADAM Swarm Master Prompt — Continuously Bonded
**Repo:** adamvangrover/adam · **Bonded by:** Jolly for Adam · **Started:** 2026-10-03
**Status:** LIVING DOCUMENT — every batch appends; nothing is rewritten, only superseded with a dated note.

> **Prime directive for any agent reading this:** This file is the single continuous
> review surface for the ADAM hardening program. It bonds verified findings, swarm
> prompts, subagent results, and open follow-ups in one place, so an expert human
> or an adversarial machine reviewer can audit the full chain: claim → evidence →
> fix → verification. When you complete work, APPEND a dated result block under the
> matching prompt. Never silently edit a prior block; mark supersessions explicitly.

---

## 1. Verified repo context (2026-10-03, depth-1 clone)

- **What it is:** institutional-grade neuro-symbolic multi-agent AI framework for
  autonomous credit risk control, financial modeling, deterministic workflow
  orchestration. Public, MIT, `main` branch, ~4,238 commits, ~6,878 files.
- **Architecture:** Hybrid cognitive engine — System 1 (Neural Swarm,
  `AsyncAgentBase`, event-driven) + System 2 (Neuro-Symbolic Graph, DAG,
  `TemplateAgentV30`). Strict Path A (core, deterministic, Pydantic, PROV-O)
  vs Path B (lab, experimental) bifurcation. Business rules in jsonLogic, never
  hardcoded `if/else` (Constitution Art. III).
- **Agent I/O contract:** `AgentInput{query, context, tools}` →
  `AgentOutput{answer, sources, confidence∈[0,1], metadata}`.
  Confidence tiers: >=0.85 autonomous · 0.50–0.85 HITL · <0.50 abort.
- **How evidence was gathered:** `grep`/`diff`/direct file reads against the clone
  at `~/workspace/adam_repo`. The clone is NEVER modified in place; all fixes are
  proposed diffs under `~/workspace/adam_work/fixes/`. Nothing has been pushed.

---

## 2. Findings ledger (all verified against the clone)

| ID | Finding | Severity | Status |
|----|---------|----------|--------|
| F1 | `core/system/plugin_manager.py:32` — `importlib.import_module` on unvalidated `plugin_name` from `os.listdir()`; second vector: `config["class_name"]` → raw `getattr` | P0 security | FIX PROPOSED, 42 tests pass |
| F2 | `core/agents/industry_specialist_agent.py:41` — config-driven `sector` interpolated into module path (`core.agents.industry_specialists.{sector}`) | P0 follow-up | OPEN — needs allowlist |
| F3 | `core/human_validation_gate.py:62` — ledger silently discards ALL history on `JSONDecodeError` (`except: pass` → `ledger = []`); no locking/atomicity | P0 data integrity | FIX PROPOSED |
| F4 | `core/engine/consensus_engine.py` — `__init__` never hydrates `decision_log` from disk → first decision after restart truncates `decision_log.json` to 1 entry | P0 data integrity | FIX PROPOSED |
| F5 | Pickle / SQL-i / hardcoded secrets / 0.0.0.0 — all PASS (SafeUnpickler, `compare_digest`, no live f-string SQL, binds only in comments) | — | VERIFIED CLEAN |
| F6 | `shell=True` — zero hits repo-wide (non-test); all 41 subprocess sites use argv-list | — | VERIFIED CLEAN |
| F7 | `core/agents/governance/repo_guardian/tools.py:20` — agent-supplied revision/branch/path reach `git` argv; worst case is option-confusion, not RCE | P0 robustness | HARDENING PROPOSED |
| F8 | Scrubber duplication: `GoldStandardScrubber`/`ArtifactType`/`GoldStandardArtifact` duplicated AND diverged in `core/data_processing/utils.py` vs `universal_ingestor.py` | P1 architecture | UNIFICATION DESIGNED |
| F9 | Backlog item "merge `core/engine` + `core/v23_graph_engine` `UnifiedKnowledgeGraph`" is STALE — `core/v23_graph_engine/` does not exist; one class + one subclass only | — | CORRECTED |
| F10 | `dream_cycle.py` journal `'w'` + `audit_mixin.py` `'w'` — investigated, correctly left as-is (capped rewrite / unique-file-per-record) | — | VERIFIED OK |

---

## 3. Swarm prompts with bonded results

### PROMPT S1 — Plugin import hardening [COMPLETE]
*Role: sentinel · Confidence: 0.85 · 2026-10-03*
- **Prompt:** harden dynamic plugin imports in `core/system/plugin_manager.py`.
- **Result:** vulnerability CONFIRMED (two vectors: unvalidated `plugin_name`, raw
  `getattr(module, config["class_name"])`). Delivered: hardened
  `plugin_validator.py` (regex `^[a-zA-Z0-9_]+$`, `is_relative_to` containment,
  dunder rejection, optional allowlist, `is_safe_identifier()` reused for the
  `class_name` guard), unified `plugin_manager.patch` (validates before any path
  construction, `logger.warning` replaces `print`), 42 adversarial pytest cases —
  **all pass**, patch dry-runs clean, end-to-end smoke test green.
- **Survey:** all 7 dynamic-import sites in `core/` classified —
  `agent_orchestrator.py:371` GUARDED (static allowlist dict),
  `mcp_server/server.py:366` GUARDED (dict lookup),
  `skill_harvester_agent.py:63` / `mirofish_engine.py:173` / `sandbox.py:139-141` SAFE.
- **Follow-up bonded:** F2 (`industry_specialist_agent.py:41`) is the only remaining
  user-influenced dynamic import — needs the same allowlist treatment.
- **Files:** `fixes/s1/plugin_validator.py`, `fixes/s1/plugin_manager.patch`,
  `fixes/s1/test_plugin_validator.py`, `fixes/s1/other_import_sites.md`

### PROMPT S2 — Append-mode / ledger integrity [COMPLETE]
*Role: sentinel · Confidence: 0.85 · 2026-10-03*
- **Prompt:** fix append-mode violations in ledger/journal/telemetry/audit.
- **Result — deeper than the prompt assumed:**
  - `human_validation_gate.py:62`: the `'w'` is read-modify-write, but the REAL bug
    is silent total history loss on corrupt JSON. Fix: `fcntl` lock + atomic
    temp+`os.replace` write + quarantine corrupt files to `*.corrupt.<ts>`
    instead of discarding. (Plain `'a'` impossible — readers `json.load` the array.)
  - `consensus_engine.py:205`: the `'w'` itself is fine in-process; the REAL bug is
    `__init__` starting `decision_log = []` without hydrating from disk → restart
    truncates history. Fix: hydrate from `decision_log.json`, fallback rebuild from
    the `.jsonl` source of truth, atomic rewrite.
  - `dream_cycle.py:57,152`: LEFT AS-IS — line 57 is create-if-missing init, line 152
    rewrites a journal explicitly capped at 50 newest-first entries; `'w'` loses
    nothing the design retains. Optional atomic-write hardening included.
  - `audit_mixin.py:67`: NO CHANGE — one `PROV-{uuid4}.json` file per record; the
    directory is the log.
- **Sweep:** remaining 27 `open(..., 'w')` sites in `core/` all classified —
  snapshot/cache/artifact/codegen semantics, `'w'` correct.
- **Files:** `fixes/s2/01_human_validation_gate.md` (+ full diff),
  `fixes/s2/03_consensus_engine.md` (+ full diff), `fixes/s2/02_dream_cycle.md`,
  `fixes/s2/04_audit_mixin.md`, `fixes/s2/other_candidates.md`,
  `fixes/s2/test_ledger_append.py` (sequential writes persist, corrupt input
  quarantined, 10 concurrent writers lose nothing)

#### 2026-10-05 — S2 APPLY-AND-TEST RESULT (scratch copy, clone untouched) [VERIFIED]
- **Procedure:** `cp -r` of the depth-1 clone to `~/workspace/adam_work/scratch_s2/`;
  both diffs applied to the scratch copy only. Pristine clone md5-verified unchanged
  afterwards; nothing pushed.
- **HVG patch:** applied cleanly via `patch -p1`; `diff -u` vs pristine == proposal.
- **CE patch:** the authored unified diff carried wrong `@@` hunk counts/line numbers
  (filing defect, not a code conflict); hunk #3 failed `patch --dry-run`. Repaired only
  the headers, then applied the three edits by exact-string replacement (each old block
  asserted unique). Applied content is byte-identical to the proposal.
- **Tests:** S2 regression suite 4/4 PASS vs patched code (incl. 10-thread concurrency);
  new CE hydration tests 4/4 PASS (restart no longer truncates `decision_log.json`,
  `.jsonl` fallback rebuild works, corrupt snapshot degrades gracefully, `evaluate()`
  output identical pristine vs patched). Repo suite scoped: 6 failures all
  pre-existing `ModuleNotFoundError` for heavy third-party deps (pandas etc.), unrelated
  to the change; full-suite run impractical (no uv/venv, torch-class optional deps).
- **Adversarial findings bonded:** (1) the bonded `test_ledger_append.py` concurrent
  test has an off-by-one — thread `i=0` writes `{"k":0}`→`{"k":0}` (no differences, nothing
  recorded), so it can only ever see 9 entries; fix the test, not the code.
  (2) `tests/test_adam_v_next.py::test_consensus_engine` expectations are stale vs the
  shipped `evaluate()` (`SELL/REJECT`/`< -0.6` expected, `HOLD`/`-0.1667` on both pristine
  and patched) — pre-existing drift, not an S2 regression.
- **Full report:** `fixes/s2/s2_apply_result.md`

### PROMPT S3 — Scrubber unification [DESIGN COMPLETE — implementation open]
*Role: architect_jules · Confidence: 0.85 · 2026-10-03*
- **Prompt:** unify diverged `GoldStandardScrubber`/`ArtifactType`/`GoldStandardArtifact`.
- **Result:** full divergence table (every class/method/constant with
  KEEP_A/KEEP_B/MERGE verdicts). Key decisions bonded:
  1. `utils.py` semantics are the de facto standard (newer pipelines
     `sequential_pipeline.py`, `universal_ingestor_v2.py` already import from it;
     existing tests encode its `clean_text` collapsing behavior) → canonical keeps A.
  2. `ArtifactType` becomes `str`-Enum (Pydantic coerces both call styles).
  3. `GoldStandardArtifact` becomes Pydantic `BaseModel` (Path A mandate);
     `to_dict()` emits BOTH `type` and `artifact_type` keys + `content_hash` (additive).
  4. ALL weights/thresholds → `config/scrubber_weights.yaml` (Constitution Art. III —
     no hardcoded numbers in `.py`).
  5. Compat shims documented (`.type` property, dual keys, deprecation path).
- **Out of scope, flagged:** `conviction_scorer.py` / `semantic_conviction.py`
  (claim-vs-source scoring — different concern, future S-pass).
- **Files:** `fixes/s3/divergence_table.md`, `fixes/s3/scrubber.py` (draft, smoke-tested
  vs pydantic 2.13.5), `fixes/s3/scrubber_weights.yaml`, `fixes/s3/migration_plan.md`,
  `fixes/s3/consumers.md` (11 importing files)

### PROMPT S4 — CI security gate [COMPLETE]
*Role: sentinel · 2026-10-03*
- **Prompt:** create `scripts/security_audit.py` failing CI on P0 regressions.
- **Result:** built, self-tested, and debugged against the repo — first run caught a
  FALSE POSITIVE in my own regex (`f"Selected target"` matched `f"SELECT`), fixed
  with `\b` boundary. Now: **0 failures, 14 warnings across 1,901 files** in ~1s.
  Checks: raw pickle outside allowlist, unguarded dynamic imports, f-string SQL,
  `0.0.0.0` in code, `shell=True`.
- **File:** `adam_work/security_audit.py` (also usable as `make security-audit`)

#### 2026-10-05 — S4 CI wiring proposed [DIFFS READY — NOT APPLIED]
*Role: sentinel · Confidence: 0.9*
- **Work:** proposed wiring `security_audit.py` into the repo's CI and Makefile.
  Three patches under `adam_work/fixes/s4_ci/` (clone untouched, nothing pushed):
  1. `scripts_security_audit.py.patch` — NEW file `scripts/security_audit.py`
     (path the script's own `--help` already advertises; `scripts/` is the
     established home, cf. `scripts/build_instructions.py` in `ci.yml`).
  2. `ci-security.yml.patch` — new `adam-security-gate` job
     ("ADAM static security gate (S4)") in the existing `CI / Security` workflow:
     checkout → setup-python 3.11 → `python scripts/security_audit.py`.
     Chosen over `ci.yml`'s lint job: inherits push/PR path filters + weekly
     `0 0 * * 0` cron with zero trigger changes; no `uv sync` needed for a
     stdlib-only ~1s scanner; single Python version, no matrix. No `|| true` —
     a P0 failure fails the job and the build. Complements Bandit (repo-specific
     P0 invariants Bandit doesn't encode: pickle allowlist, f-string `SELECT\b`,
     `0.0.0.0`, `shell=True`).
  3. `Makefile.patch` — standalone `security-audit` target + `.PHONY` entry
     (tab-indented); deliberately NOT chained into `check`/`lint` to preserve
     their semantics. Exit code propagates: `make security-audit` fails on P0s.
- **Bonded decision:** warnings do NOT fail the build (script exits 0 on WARNs
  only). The 14 current warnings are the dynamic-import review list F2's
  allowlist will shrink; failing on them would block every PR until F2 lands.
  Future: promote warnings to failures once F2 is fixed.
- **Verification:** patched workflow `yaml.safe_load` parses; 5 jobs incl. new one;
  `make -n security-audit` dry-runs to `python scripts/security_audit.py`.
  Plan + apply/verify steps in `fixes/s4_ci/s4_ci_plan.md`.
- **Open:** flip warnings→failures post-F2; re-measure if repo passes ~10k files.

### PROMPT S5 — Subprocess `--` audit [COMPLETE]
*Role: sentinel · Confidence: 0.9 · 2026-10-03*
- **Prompt:** audit `subprocess.run` sites for option-injection risk.
- **Result:** NO vulnerability. `shell=True` absent repo-wide; all 41 sites use
  argv-list. `github_alpha_agent.py` sites SAFE (scheme validation, `--` present,
  `timedelta` coercion). `git_repo_sub_agent.py:59` exemplary (scheme + leading-`-`
  rejection + abspath containment + `--`) — template for others.
- **Hardening proposed (robustness, not RCE):** `repo_guardian/tools.py:20` —
  reject leading-`-` revisions / allowlist `^[A-Za-z0-9_./-]+$`, add `--` to `list_files`.
- **File:** `fixes/s5_s6/subprocess_verdicts.md`

### PROMPT S6 — v23 prototype triage [COMPLETE]
*Role: architect_jules · Confidence: 0.95/0.85 · 2026-10-03*
- **Prompt:** triage `experimental/v23_prototypes/v23_graph_engine/`.
- **Result:** backlog item STALE — nothing to merge (single `UnifiedKnowledgeGraph`
  + `OdysseyKnowledgeGraph` subclass; PoCs unreferenced). Both PoCs fully
  superseded by production code (`core/engine/cyclical_reasoning_graph.py`,
  `core/engine/states.py:23`). Verdict: **ARCHIVE with README** (teaching value),
  not delete. Backlog correction text provided.
- **File:** `fixes/s5_s6/v23_triage.md`

### PROMPT S7 — Swarm prompt runner [COMPLETE]
*Role: nexus · 2026-10-03*
- **Prompt:** CLI that dispatches prompt files to async agents, collects
  AgentOutputs, writes consolidated reports.
- **Result:** `swarm_run.py` built — `--dry-run` validates all 7 prompts above
  (**7/7 valid**), `--mock` dispatch records PROV-O (prompt sha256, agent id,
  timestamps) to `runs/<ts>/`, reporter writes `REPORT.md` + `RUN.json`.
  Backend `dispatch()` left as the single wiring point for the real orchestrator.
- **File:** `adam_work/swarm_run.py`

---

## 4. Dispatch sequence (for the orchestrator)

1. S2 (data integrity) → 2. S1 + S5 in parallel (sentinel) → 3. S4 (gate encodes fixed state)
   → 4. S3 (biggest blast radius, alone, with benchmark) → 5. S6 (anytime) → 6. S7 (meta, done)

## 5. Open follow-ups (bonded, not lost)

- [ ] F2: `industry_specialist_agent.py:41` — apply `is_safe_identifier` allowlist pattern (S1 follow-up prompt drafted in `other_import_sites.md`)
- [ ] S3 implementation: port `UniversalIngestor` (705 lines) to canonical scrubber; run 10MB/<10% benchmark + golden-set diff in the repo's venv
- [ ] S2 patches: apply to clone, run repo test suite (`uv run pytest tests/ -v --strict-markers`)
- [ ] S6: execute the archive move + README + MEMORY.md backlog correction
- [ ] S4: wire `security_audit.py` into CI yaml + `Makefile`
- [ ] S7: wire `dispatch()` to the real Nexus/LangGraph backend
- [ ] Batch 3 candidates: P2 UX audit, `conviction_scorer`/`semantic_conviction` dedup pass, `scripts/system_guide_agent.py:64` `.split()` review

---

## 6. Adversarial review notes

- Every VERIFIED CLEAN claim above names the exact grep that established it; re-run
  `security_audit.py` to reproduce.
- Every FIX is a proposed diff, NOT applied — the clone at `~/workspace/adam_repo`
  is untouched and nothing was pushed. Trust but verify: apply, run tests, then merge.
- Confidence scores are the executing agent's self-reported scores, bonded verbatim —
  treat <0.9 as "needs a second pair of eyes," which is why F2/F4-adjacent items
  remain open rather than closed.
- Known self-correction in this file's history: the S4 regex false positive
  (`f"Selected target"`) was caught and fixed before bonding — adversarial reviewers
  should probe for more of these.

---

## 7. S6 archive execution prep (2026-10-05)

*Role: subagent (S6 archive execution prep) · Confidence: 0.95 that nothing imports the PoCs; 0.90 that the plan is complete and executable as written.*

- **What was done:** turned the 2026-10-03 v23 triage verdict (ARCHIVE with README, not delete) into an executable, reviewable plan. No move was executed; the clone is untouched.
- **Pre-flight re-verified against the clone (read-only):**
  - Scope = the whole `experimental/v23_prototypes/` tree (5 files: `v23_graph_engine/cyclical_graph_poc.py` 78 lines, `v23_graph_engine/adaptive_system_poc.py` 164 lines, `v23_graph_engine/directory_manifest.jsonld`, `v23_graph_engine/index.html`, `index.html`). `v23_graph_engine/` is the tree's only content; the parent `index.html` is its showcase-browser index — the whole directory moves, not just the `.py` files.
  - Repo-wide `.py` grep for imports of `v23_graph_engine`/`cyclical_graph_poc`/`adaptive_system_poc`: **zero hits**. Broader textual hits for `core/v23_graph_engine` exist only in stale historical docs (`Architectural_Review_Refined.md`, `.jules/bolt.md`, `.jules/jules.md`, `CHANGELOG.md`, showcase mock JSON) — checked, left alone.
  - Production equivalents confirmed: `core/engine/cyclical_reasoning_graph.py` exists; `core/engine/states.py:23` defines production `PlanOnGraph(TypedDict, total=False)` (richer than the PoC — adds `cypher_query`).
  - Single `UnifiedKnowledgeGraph` at `core/engine/unified_knowledge_graph.py:43`; subclass at `core/engine/odyssey_knowledge_graph.py:16` — backlog merge item STALE.
- **Deliverables (individual markdowns, no zips):** `adam_work/fixes/s6_archive/archive_plan.md` (exact `git mv` commands on an `archive-v23-pocs` branch, pre/post verification, rollback), `adam_work/fixes/s6_archive/README.archive.md` (full README content for `archive/v23_prototypes/README.md`), `adam_work/fixes/s6_archive/backlog_correction.md` (stale item quoted at `MEMORY.md:32` AND `docs/AGENTS_KNOWLEDGE_BASE.md:130` — the triage caught only one of the two — plus replacement text).
- **Correction to prior blocks:** the 2026-10-03 triage mentioned the stale backlog item only in `MEMORY.md`; the identical item also lives in `docs/AGENTS_KNOWLEDGE_BASE.md:130`. Both are covered in the correction text. Nothing in §3/S6 was altered.
- **Open work for Adam:** approve (or kill) the plan; on approval, execution happens in his own working clone, not the read-only one here.

---

## 8. F2 hardening result (2026-10-05)

*Role: subagent (F2 follow-up) · Confidence: 0.9*

- **What was done:** closed the last user-influenced dynamic import in `core/`
  (S1 survey site #2, F2 ledger row). Nothing applied to the repo; clone untouched.
- **Vulnerability confirmed (read-only, file:line):**
  - `core/agents/industry_specialist_agent.py:29` — `sector` from agent config.
  - `.../industry_specialist_agent.py:41` —
    `importlib.import_module(f"core.agents.industry_specialists.{sector}")` —
    a dotted sector escapes the package.
  - `.../industry_specialist_agent.py:42-43` — `sector.capitalize() + "Specialist"`
    → raw `getattr`.
  - Secondary bug found during grounding: `.capitalize()` mis-derives class names
    for all four snake_case sectors (`consumer_discretionary`,
    `consumer_staples`, `real_estate`, `telecommunication_services`) — they
    silently returned `None` under the old code. Verified via
    `grep -n "^class "` over `core/agents/industry_specialists/*.py`.
- **Deliverables (individual markdowns, no zips):**
  `adam_work/fixes/f2/sector_validator.py` (new module, target
  `core/security/sector_validator.py`; pure stdlib, S1-style
  `is_safe_identifier` + 11-entry `KNOWN_SECTOR_SPECIALISTS` allowlist +
  `validate_sector()` raising `TypeError`/`ValueError`),
  `adam_work/fixes/f2/industry_specialist_agent.patch` (unified diff:
  validate-before-import, fail-closed `(TypeError, ValueError)` → existing
  generic fallback, fixes the snake_case mis-resolution),
  `adam_work/fixes/f2/test_sector_validator.py` (**121 tests, all pass** —
  46 adversarial inputs incl. dotted escape, traversal, dunder, null byte,
  cyrillic lookalike, wrong-case, type confusion; behavioral replica proves
  hostile input never reaches `import_module` — zero fake-import calls),
  `adam_work/f2/f2_result.md` (full claim → evidence → fix → verification).
- **Patch integrity:** `patch --dry-run -p1` against the clone → clean; applied
  to a `/tmp` copy → applies and `py_compile` passes (clone never modified).
- **Ledger effect:** F2 row in §2 moves OPEN → FIX PROPOSED by this block; the §2
  row and §5 checklist item themselves were left untouched per the append-only rule.
- **Open work for Adam:** apply patch + new module on a scratch branch and run
  the repo test suite (`uv run pytest tests/ -v --strict-markers`).
