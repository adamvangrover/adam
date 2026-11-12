# Build Guide: Autonomous Multi-Agent Pipeline & Deterministic Subsystems

## 1. Scope & Objective
This Build Guide provides actionable instructions for AI agents and human systems engineers constructing high-reliability, neuro-symbolic agent pipelines within the ADAM / AFOS monorepo. It details end-to-end implementation from AST scaffolding to Rust PDIL gatekeeper hooks and CBF safety clamping.

## 2. Monorepo Structural Rules & Path Bifurcation
- **Path A (Strict Institutional Core)**: `core/`, `src/`, `adam_*/`, `core_kernel/`.
  - Zero UI dependencies allowed (no Streamlit, no React).
  - Strict Python type annotations required (enforce `dict` over `typing.Dict`).
  - Mandatory zero-copy parsing (`pydantic.TypeAdapter.validate_json`).
- **Path B (Research & Experimental Sandboxes)**: `experimental/`, `research/`, `tinker_lab/`.
  - Path A must never import from Path B.
- **Native Extension Integration**: Rust extensions reside in `rust_ext/` and build via `maturin`.

## 3. Step-by-Step Architecture Scaffolding Protocol

### Step 1: Interface & Schema Declaration
1. Define deterministic Pydantic models in `core/schemas/` or `src/schemas/`.
2. Ensure every output type inherits or includes `ProvenanceHeader` (`content_hash`, `timestamp`, `agent_id`, `version`).
3. Include explicit bounding checks:
   - Probabilities $\in [0.0, 1.0]$.
   - Spreads $\ge 0.0\text{ bps}$.
   - Confidence floor $\ge 0.85$ for autonomous execution.

### Step 2: Probabilistic-to-Deterministic Integration Layer (PDIL)
1. Wrap probabilistic LLM inferences in `src/pdil/middleware.py`.
2. Connect the `SecurityGovernanceGatekeeper` to inspect:
   - Multiplicative Layer 0 safety checks.
   - Grounding validation via `check_grounding()`.
3. Configure `DriftIntelligenceLayer` with observed drift flags to trigger self-healing.

### Step 3: Concurrency & Async Scheduling
1. Use `BoundedDAGScheduler` or `asyncio.Semaphore` task groups to prevent thread exhaustion.
2. In high-throughput JSON validation loops, compile the schema validator once outside the loop using `jsonschema.validators.validator_for(schema)`.
3. In Rust FFI bindings, ensure GIL release via `py.allow_threads` during computational or I/O bounds.

### Step 4: Verification & Property-Based Testing
1. Author unit tests under `tests/unit/`.
2. Integrate Hypothesis strategies for edge cases (NaN, $\pm\infty$, empty arrays, unicode injection).
3. Verify test coverage passes the strict minimum standard:
   ```bash
   uv run pytest tests/unit/test_[module].py
   ```

## 4. Anti-Patterns & Common Pitfalls
| Pitfall | Root Cause | Mandated Remediation |
| :--- | :--- | :--- |
| `df.iterrows()` | High CPU overhead & non-vectorized execution | Use vectorized operations or `.to_dict(orient='records')` |
| Direct artifact edits | Editing `dist/` or `target/` directly | Trace back to source files and recompile |
| Eval hacking | Tweaking verifier assertions to pass tests | Fix the actual pipeline logic, keep evaluation criteria pure |
| Amnesia / Log truncate | Using `w` mode for operational journals | Always use append-only mode: `open(f, 'a')` |
