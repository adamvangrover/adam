# The Adam Agent Knowledge Base (Consolidated Memory)

> **WARNING TO ALL AGENTS:** This document contains the "Pheromones" left by previous iterations of the system.
> **MANDATORY:** Before writing any code, search this document for relevant keywords (e.g., `graph`, `pickle`, `security`, `frontend`, `log`).
> **Directives herein supersede standard coding habits.**

---

## 🛡️ Security (Sentinel's Watch)

### 🚨 Critical Vulnerabilities (P0)

1.  **Insecure Deserialization (`pickle`)**
    *   **Finding:** `pickle.load()` allows Arbitrary Code Execution (RCE).
    *   **Status:** Partially Fixed. `core/analysis/technical_analysis.py` uses `safe_unpickler`, but other areas may be exposed.
    *   **Directive:** **NEVER** use `pickle.load()`. Use `core.security.safe_unpickler.safe_load()` or safer formats like JSON/ONNX.

2.  **Dynamic Import RCE**
    *   **Finding:** Endpoints like `/api/simulations/<name>` using `importlib.import_module(name)` allow RCE.
    *   **Directive:** **NEVER** pass user input directly to `import_module`. Use explicit sector mapping (`core/security/sector_validator.py`) or plugin validation (`core/security/plugin_validator.py`).

3.  **Path Traversal & SSRF**
    *   **Finding:** Agents reading files or fetching URLs based on user input are vulnerable.
    *   **Directive:**
        *   Files: Use `os.path.abspath` + `startswith(SAFE_ROOT)` or `Path.is_relative_to`.
        *   URLs: Validate scheme (`https`), block private IPs (`10.0.0.0/8`, `127.0.0.1`), and use `ssl.create_default_context()`.

4.  **SQL Injection**
    *   **Finding:** Dynamic f-string SQL queries pose severe injection risk.
    *   **Directive:** Use parameter substitution (`con.execute(query, parameters=params)` in DuckDB/SQLite). **NEVER** construct SQL strings with `f-strings`.

5.  **Log Leakage**
    *   **Finding:** Raw log endpoints expose sensitive tokens and context states.
    *   **Directive:** **NEVER** expose internal logs via public APIs. Use structured, sanitized response objects.

---

## ⚡ Architecture & Performance (Bolt's Optimization)

### 🏗️ Structural Traps (P1)

1.  **Duplicate Knowledge Graph Classes**
    *   **Finding:** `core/engine/unified_knowledge_graph.py` and legacy graph engines.
    *   **Directive:** Prefer `core/engine/cyclical_reasoning_graph.py` and `core/engine/states.py`. Legacy PoCs are archived under `archive/v23_prototypes/`.

2.  **Duplicate Logic (Scrubbers & Ingestors)**
    *   **Finding:** `core/data_processing/utils.py` holds standard `GoldStandardArtifact` and `GoldStandardScrubber` with YAML scrubber weight configs (`config/scrubber_weights.yaml`).
    *   **Directive:** Use `GoldStandardScrubber` in `core/data_processing/utils.py`.

### 🚀 Performance (P2)

1.  **PyO3 GIL Releases in Rust Kernel**
    *   **Finding:** Native Monte Carlo and VaR calculations in `rust_ext` must release Python GIL.
    *   **Directive:** Wrap heavy C-level/Rust computations in `py.allow_threads(|| { ... })` and export PyO3 0.23+ bindings with `rlib` feature flags.

2.  **Clean-Room Evaluation & Offline Replay**
    *   **Finding:** LLM-as-a-Judge evaluations require zero external network dependencies for reproducible benchmark passes.
    *   **Directive:** Use `CleanRoomLLMJudge` in `evals/graders/llm_judge.py` with multi-criteria weighted rubrics and offline replay key mappings.

---

## 🎨 UX & Accessibility (Palette's Standards)

### ♿ Accessibility & Presentation Layer (P2)

1.  **Presentation Boundaries**
    *   **Directive:** Distinguish Streamlit internal tools from React/Vite web apps (`services/webapp/` or `frontend/`) and static benchmark showcases (`showcase/`).

2.  **Keyboard & ARIA Standards**
    *   **Directive:** Terminals must include `role="log"` and `aria-live="polite"`. Loading state indicators must use `role="status"`.

---

## 🐝 Swarm Intelligence (Async Operations)

### ⚠️ Concurrency Hazards (P1)

1.  **Cross-Agent Communication**
    *   **Directive:** Agents must NOT instantiate other agents directly. Always emit structured metadata routing requests via `Nexus Orchestrator` to preserve linear W3C PROV-O audit trails.

2.  **File Overwrites & Process Locking**
    *   **Directive:** Combine `fcntl.flock` file locks with `threading.Lock()` to ensure both inter-process and intra-process safety for JSON ledgers.

---

## 📋 Prioritized Backlog (Incoming Requests)

### P0: Security Hardening (Sentinel)
- [x] **Modernize PyO3 & Rust Extensions:** Upgraded `rust_ext` with PyO3 0.23, GIL releases, and `rlib` feature gating.
- [ ] **Audit `importlib` Usage:** Grep for dynamic imports and enforce strict mapping filters.
- [ ] **Fix SQL Injection:** Verify parameter binding across all DuckDB/SQLite query call sites.

### P1: Architectural Refactoring & Evaluation (Bolt)
- [x] **Deduplicate Scrubbers:** Standardized `GoldStandardArtifact` in `core/data_processing/utils.py`.
- [x] **Multi-State Reasoning Evaluator:** Integrated `evals/graders/multi_state_reasoning_evaluator.py`.
- [x] **Clean-Room LLM-as-a-Judge:** Enhanced `evals/graders/llm_judge.py` with multi-criteria rubrics and offline replay store.

### P2: UX & Context Management (Palette & Nexus)
- [x] **Agent Catalog Expansion:** Expanded `AGENT_CATALOG.md` with detailed agent cards for all 35+ system personas.
- [x] **Context Deduplication:** Cleaned `AGENTS.md`, `llms.txt`, and `llms-full.txt` to eliminate conversational prompt leakage and duplicate version generations.
