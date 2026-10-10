# TASK INGESTION MANIFEST: ADAM REPO HARDENING & AUTONOMOUS DISPATCH
**Repo Target:** `adamvangrover/adam` · **Branch:** `main`
**Execution Context:** Sandboxed Multi-Agent Swarm / Compute Subagent Runner
**Audit Standard:** Continuous PROV-O Lineage · Strictly Zero In-Place Edits on Base Clone
**Operating Rule:** Append-only reporting · Fail-closed security · Constitution Art. III (no hardcoded logic in `.py`)

---

## 0. AUTONOMOUS RUNNER DIRECTIVE & ENVIRONMENT TOPOLOGY

Any autonomous agent, subagent, or batch compute container ingesting this manifest must adhere to the following sandbox rules:

1. **Isolation:** Never touch `/workspace/adam_repo` directly. Always mirror the target tree to a transient execution scratchpad:
   ```bash
   cp -r /workspace/adam_repo /workspace/adam_work/scratch_run
   cd /workspace/adam_work/scratch_run
   ```
2. **Artifact Lineage:** Every fix must be emitted as an atomic patch file under `/workspace/adam_work/fixes/<task_id>/` along with a companion test suite verifying the change.
3. **Execution Verification:** Run scoped pytest suites within the project's dependency environment:
   ```bash
   uv run pytest tests/ -v --strict-markers
   ```
4. **Append-Only Ledger:** Upon task completion, append your execution block to the Master Review Log with schema: `[TASK_ID] · [ROLE] · [CONFIDENCE 0.0-1.0] · [OUTCOME] · [EVIDENCE/HASHES]`

---

## 1. FINDINGS LEDGER & EXECUTION STATUS

| ID | Module Target | Vector / Vulnerability | Target State | Severity | Status |
|---|---|---|---|---|---|
| F1 | `core/system/plugin_manager.py` | `importlib.import_module` on unvalidated `plugin_name` via `os.listdir()`; unescaped `getattr(module, class_name)`. | Safe identifier regex (`^[a-zA-Z0-9_]+$`), path containment check (`is_relative_to`), dunder block, safe `getattr`. | P0 Security | PATCH READY (S1) |
| F2 | `core/agents/industry_specialist_agent.py` | Dotted sector config parameter escapes package via `importlib.import_module(f"...{sector}")`. `.capitalize()` breaks `snake_case` classes. | Implement `core/security/sector_validator.py` with static 11-sector allowlist and explicit class name map. | P0 Security | PATCH READY |
| F3 | `core/human_validation_gate.py` | Line 62: Silent total history purge on `JSONDecodeError` (`except: pass -> ledger = []`). No file locking. | Read-modify-write wrapped in `fcntl.flock`, atomic temp write + `os.replace`, quarantine corrupt logs to `*.corrupt.<ts>`. | P0 Data Integrity | VERIFIED (S2) |
| F4 | `core/engine/consensus_engine.py` | `__init__` does not rehydrate `decision_log` from disk; first post-restart decision truncates log array to 1 entry. | Boot hydration from `decision_log.json`, fallback rebuild from `.jsonl` audit log, atomic snapshot rewrite. | P0 Data Integrity | VERIFIED (S2) |
| F5 | Core Codebase | Audit for pickle, SQL injection, hardcoded secrets, 0.0.0.0 bindings. | Continuous static gate verification (`scripts/security_audit.py`). | Invariant | CLEAN |
| F6 | Subprocess Execution | Audit for `shell=True` and option injection across 41 subprocess sites. | Enforce argv-list everywhere; inject `--` argument delimiters where user input reaches arguments. | Invariant | CLEAN |
| F7 | `core/agents/governance/repo_guardian/tools.py` | Agent-supplied branch/path reach `git` CLI argv without leading-dash guards. | Option-confusion hardening: reject revisions starting with `-`, regex validation `^[A-Za-z0-9_./-]+$`. | P0 Robustness | PROPOSAL READY (S5) |
| F8 | `core/data_processing/utils.py` vs `universal_ingestor.py` | `GoldStandardScrubber` diverged across modules. Hardcoded weights in `.py`. | Unify to canonical Pydantic model (`GoldStandardArtifact`), externalize weights to `config/scrubber_weights.yaml`. | P1 Architecture | SPEC COMPLETE (S3) |
| F9 | `experimental/v23_prototypes/` | Stale v23 prototypes superseded by `core/engine/cyclical_reasoning_graph.py`. | Relocate to `archive/v23_prototypes/` with contextual README; fix MEMORY.md and KB docs. | Tech Debt | PLAN READY (S6) |

---

## 2. DISPATCH BATCH SPECIFICATIONS (FOR SUBAGENT INGESTION)

### DISPATCH UNIT 1: S1 + F2 — Dynamic Import Hardening
* **Objective:** Eliminate untrusted dynamic module loading and arbitrary `getattr` execution.
* **Action Items:**
    1. Write `core/security/plugin_validator.py`:
        * Assert identifier conforms to `^[a-zA-Z0-9_]+$`.
        * Reject double-underscores (`__`).
        * Validate resolved filesystem path is child of plugin directory using `pathlib.Path.is_relative_to`.
    2. Apply `fixes/s1/plugin_manager.patch`:
        * Validate `plugin_name` before constructing path.
        * Validate `class_name` using `is_safe_identifier` before calling `getattr`.
        * Replace raw prints with `logger.warning`.
    3. Write `core/security/sector_validator.py`:
        * Define `KNOWN_SECTOR_SPECIALISTS` mapping 11 canonical sectors (financials, technology, consumer_discretionary, etc.) to explicit class names.
        * Implement `validate_sector(sector: str) -> Tuple[str, str]` raising `ValueError`/`TypeError` on unlisted inputs.
    4. Apply `fixes/f2/industry_specialist_agent.patch`:
        * Route sector ingestion through `validate_sector`.
        * Wrap dynamic import in `(TypeError, ValueError)` catch block falling back to generic specialist.
* **Verification Harness:**
    1. Execute `fixes/s1/test_plugin_validator.py` (42 tests).
    2. Execute `fixes/f2/test_sector_validator.py` (121 tests including directory traversal, null-byte injection, and casing checks).

### DISPATCH UNIT 2: S2 — Ledger & State Integrity (Verification & Landing)
* **Objective:** Prevent data loss from concurrency races and corrupt ledger files.
* **Action Items:**
    1. Patch `core/human_validation_gate.py`:
        * Wrap ledger file opens in exclusive `fcntl.flock`.
        * Implement atomic file replacement: write to `.tmp.<pid>` then `os.replace`.
        * If `json.load` encounters `JSONDecodeError`, copy raw corrupted file to `ledger.json.corrupt.<timestamp>` before re-initializing ledger.
    2. Patch `core/engine/consensus_engine.py`:
        * In `__init__`, load existing records from `decision_log.json` if present.
        * If snapshot is missing or corrupt, rebuild state by reading chronological records from `.jsonl`.
* **Verification Harness:**
    1. Run `fixes/s2/test_ledger_append.py`.
    2. Validate 10-thread concurrent writer test (verify fix for index-0 diff recording).
    3. Assert zero record truncation on simulated interpreter restart.

### DISPATCH UNIT 3: S4 — CI Static Security Gate Wiring
* **Objective:** Enforce zero P0 regressions at build time.
* **Action Items:**
    1. Deploy `scripts/security_audit.py` to repo root:
        * Regex scan for: unpickling outside safe list, raw `f"SELECT\b"`, `0.0.0.0` bindings, `shell=True`, and dynamic imports lacking validator wrappers.
        * Exit code 1 on any P0 invariant breach; exit code 0 with warning logs for pending migrations.
    2. Apply `Makefile.patch`:
        * Register `.PHONY: security-audit` target: `python scripts/security_audit.py`.
    3. Apply `ci-security.yml.patch`:
        * Add `adam-security-gate` job running on Python 3.11 without external heavy dependencies.
* **Verification Harness:**
    1. Run `make security-audit` against clean repo (must exit 0 with <= 14 warnings).
    2. Run synthetic negative test: inject `eval(user_input)` or `f"SELECT * FROM {table}"` and confirm gate exit code 1.

### DISPATCH UNIT 4: S5 + S6 — Git Option Injection & Archive Cleanout
* **Objective:** Eliminate argument injection in repo tooling and decommission dead experimental code.
* **Action Items:**
    1. Patch `core/agents/governance/repo_guardian/tools.py`:
        * Reject git references starting with `-`.
        * Add `--` delimiter between options and target paths: `git diff <rev> -- <path>`.
    2. Execute v23 Archive Migration:
        * Run: `git mv experimental/v23_prototypes archive/v23_prototypes`
        * Commit `archive/v23_prototypes/README.md` explaining production supersession by `core/engine/cyclical_reasoning_graph.py`.
        * Update backlog references in `MEMORY.md` and `docs/AGENTS_KNOWLEDGE_BASE.md`.
* **Verification Harness:**
    1. Verify repo imports: `git grep "v23_graph_engine"` returns zero hits in active runtime modules.
    2. Test `repo_guardian` tools against hostile branch names (e.g., `--upload-pack=...`).

### DISPATCH UNIT 5: S3 — Scrubber Unification Architecture
* **Objective:** Eliminate redundant scrubber implementations and align with Pydantic Path A standard.
* **Action Items:**
    1. Externalize all cleaning heuristics, thresholds, and regex weights to `config/scrubber_weights.yaml`.
    2. Implement unified `GoldStandardArtifact` inheriting from `pydantic.BaseModel`:
        * Add `.to_dict()` backward compatibility shim exposing both `type` and `artifact_type`.
        * Implement SHA-256 `content_hash` generation on text ingest.
    3. Migrate `universal_ingestor.py` callers to import from `core/data_processing/scrubber.py`.
* **Verification Harness:**
    1. Run 10MB document throughput benchmark: memory delta must remain < 10%.
    2. Run golden dataset regression tests ensuring byte-exact normalization parity with legacy `utils.py` outputs.

---

## 3. PROV-O AUDIT & HANDOFF VERIFICATION TEMPLATE
When a worker agent finishes a batch, append the following record to the active tracking file:

```markdown
### [BATCH_RUN_<TIMESTAMP>] — Task: <TASK_ID>
- **Executing Agent ID:** <agent_uuid_or_role>
- **Source Commit SHA:** <git_rev_parse_head>
- **Target Files Touched:**
  - `path/to/file1.py` (md5: <hash>)
  - `path/to/file2.patch` (md5: <hash>)
- **Test Invariants Checked:**
  - [x] Syntax check (`py_compile`) passed
  - [x] Targeted test harness passed (<passed_count>/<total_count>)
  - [x] Base repo clone untouched (verified via git status)
- **Status:** [COMPLETE | BLOCKED | SUPERSEDED]
- **Operational Notes / Edge Cases Observed:**
  <brief narrative of findings or downstream impacts>
```
