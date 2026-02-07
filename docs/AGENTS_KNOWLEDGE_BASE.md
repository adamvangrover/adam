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
    *   **Directive:** **NEVER** pass user input directly to `import_module`. Use a strict whitelist or dictionary mapping.

3.  **Path Traversal & SSRF**
    *   **Finding:** Agents reading files or fetching URLs based on user input (e.g., `GitRepoSubAgent`, `SupplyChainRiskAgent`) are vulnerable.
    *   **Directive:**
        *   Files: Use `os.path.abspath` + `startswith(SAFE_ROOT)`.
        *   URLs: Validate scheme (`https`) and block private IPs (`10.0.0.0/8`, `127.0.0.1`, etc.).

4.  **SQL Injection**
    *   **Finding:** Blacklisting keywords (`DROP`) is insufficient.
    *   **Directive:** Use `sqlite3.set_authorizer` or ORM features. **NEVER** construct SQL strings with `f-strings`.

5.  **Log Leakage**
    *   **Finding:** `/api/state` exposed raw server logs containing secrets.
    *   **Directive:** **NEVER** expose internal logs via public APIs. Use structured, sanitized response objects.

### ⚠️ High Risk (P1)

1.  **Hardcoded Secrets**
    *   **Finding:** Fallbacks like `os.environ.get('KEY') or 'default'` leak in production.
    *   **Directive:** Fail fast. `os.environ['KEY']` (raise KeyError) is better than a weak default.

2.  **Flask Misconfiguration**
    *   **Finding:** Default Flask leaks stack traces and allows all CORS.
    *   **Directive:** Explicitly configure `CORS(resources={...})` and register a generic error handler for `500` errors.

---

## ⚡ Architecture & Performance (Bolt's Optimization)

### 🏗️ Structural Traps (P1)

1.  **Duplicate Knowledge Graph Classes**
    *   **Finding:** `core/engine/unified_knowledge_graph.py` and `core/v23_graph_engine/unified_knowledge_graph.py` are distinct but similar.
    *   **Status:** **ACTIVE RISK.**
    *   **Directive:** Always check imports. Prefer `core/v23_graph_engine/` (check latest `README` or `AGENTS.md` for current standard). **Do not edit one without checking the other.**

2.  **Duplicate Logic (Scrubbers)**
    *   **Finding:** `core/data_processing/utils.py` and `core/data_processing/universal_ingestor.py` duplicate text cleaning logic.
    *   **Directive:** Refactor to a shared utility in `core/common/text_utils.py` before modifying either.

### 🚀 Performance (P2)

1.  **Graph Loading Bottleneck**
    *   **Finding:** Re-instantiating `UnifiedKnowledgeGraph` parses JSON every time.
    *   **Directive:** Use a **Singleton** pattern or module-level caching for read-only graph structures.

2.  **React Re-Renders**
    *   **Finding:** Terminals re-render entire history on every keystroke.
    *   **Directive:** Separate `InputState` (fast) from `HistoryState` (slow) into different components.

3.  **Database Connections**
    *   **Finding:** Initializing Neo4j drivers inside request handlers kills performance.
    *   **Directive:** Initialize drivers **once** at app startup (`g` object in Flask or global singleton).

4.  **NumPy Vectorization**
    *   **Finding:** Python loops for Monte Carlo are 8x slower.
    *   **Directive:** Always use `numpy` vector operations for simulations.

---

## 🎨 UX & Accessibility (Palette's Standards)

### ♿ Accessibility (P2)

1.  **Terminal/Log Output**
    *   **Directive:** Use `role="log"` and `aria-live="polite"` for dynamic content updates.

2.  **Keyboard Navigation**
    *   **Directive:**
        *   Global Search: `Ctrl+K` shortcut.
        *   Scrollable Areas: `tabIndex="0"`.
        *   Custom Dropdowns: `role="combobox"` with `aria-activedescendant`.

3.  **Loading States**
    *   **Directive:** Spinners must have `role="status"` and invisible text ("Loading...") for screen readers.

### 🛠️ Frontend Logic (P2)

1.  **Raw Data Editors**
    *   **Finding:** Binding JSON editors directly to state breaks typing.
    *   **Directive:** Use a local "draft" string state and parse on blur/submit.

---

## 🐝 Swarm Intelligence (Async Operations)

### ⚠️ Concurrency Hazards (P1)

1.  **File Overwrites ("Flash-Memory Amnesia")**
    *   **Finding:** Agents using `open(f, 'w')` destroy history.
    *   **Directive:** **ALWAYS** use `open(f, 'a')` for logs/journals unless strict replacement is required. Read the file first to understand context.

2.  **Agent Communication**
    *   **Finding:** Circular dependencies when Agents instantiate Agents.
    *   **Directive:** Use the `MetaOrchestrator`. Return a *request* for another agent to run, don't run it yourself.

---

## 📋 Prioritized Backlog (Incoming Requests)

> **Task for Agents:** Pick items from this list based on Priority (P0 = Highest).

### P0: Security Hardening (Sentinel)
- [ ] **Audit `importlib` Usage:** Grep for dynamic imports and restrict them.
- [ ] **Fix SQL Injection:** Replace all `f"SELECT ... {var}"` with parameterized queries.
- [ ] **Secure `pickle`:** Verify no other `pickle.load` exist outside `technical_analysis.py`.
- [ ] **API Auth:** Implement Middleware for `/api/agents` and `/api/simulations`.

### P1: Architectural Refactoring (Bolt)
- [ ] **Merge Graph Classes:** Consolidate `core/engine` and `core/v23_graph_engine` versions of `UnifiedKnowledgeGraph`.
- [ ] **Deduplicate Scrubbers:** Merge `utils.py` and `universal_ingestor.py` logic.
- [ ] **Fix Async Loggers:** Ensure all Swarm agents use append mode.

### P2: UX Improvements (Palette)
- [ ] **Accessibility Audit:** Run a linter/audit on `services/webapp/client` for `aria-` attributes.
- [ ] **Debounce Inputs:** Check all `onChange` handlers for range sliders/search inputs.

### P3: Documentation & housekeeping
- [ ] **Update Tutorials:** Ensure `docs/tutorials/` reflect these new mandates.
