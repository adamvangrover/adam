# The ADAM Agent Knowledge Base & Consolidated Institutional Memory (v30.1 / AFOS V3)

> **WARNING TO ALL AGENTS (`architect_jules`, `sentinel`, `bolt`, `palette`, `underwriting_agent`, `nexus`):**
> This document contains the consolidated "Pheromones" and Bitemporal Institutional Memory synthesized across all historical AFOS iterations and Google Jules autonomous engineering sessions (`adamvangrover/adam`).
>
> **MANDATORY PRE-FLIGHT:** Before writing code, modifying schemas, or authoring credit/macro intelligence, search this document for relevant domain directives (`zero_trust`, `pickle`, `ssrf`, `entmax`, `graph`, `pandas`, `react`, `lease_capitalization`, `prov_o`, `security`, `frontend`, `log`, `pdil`, `harness2`, `fortress`). Directives herein supersede generic coding habits.

---

## Table of Contents
1. [Architectural & Governance Invariants (AFOS v30.1 & V3 Kernel)](#1-architectural--governance-invariants-afos-v301--v3-kernel)
2. [Sentinel's Watch: Security & Zero-Trust Hardening (2024–2026 Full Ledger)](#2-sentinels-watch-security--zero-trust-hardening-20242026-full-ledger)
3. [Bolt's Journal: Architecture, Performance, Vectorization & Concurrency](#3-bolts-journal-architecture-performance-vectorization--concurrency)
4. [Palette's Standards: UX, Accessibility (WCAG/ARIA) & Cyberpunk UI](#4-palettes-standards-ux-accessibility-wcagaria--cyberpunk-ui)
5. [Swarm Intelligence, PDIL, Harness 2 & Kernel Architecture (ADAM-IAK-CORE)](#5-swarm-intelligence-pdil-harness-2--kernel-architecture-adam-iak-core)
6. [Quantitative Credit Forensics, Market Contagion & Multi-Horizon Memory](#6-quantitative-credit-forensics-market-contagion--multi-horizon-memory)
7. [Jules CLI (@google/jules), Local SDK & Autonomous Workflow Memory](#7-jules-cli-googlejules-local-sdk--autonomous-workflow-memory)
8. [Prioritized Engineering Backlog](#8-prioritized-engineering-backlog)

---

## 1. Architectural & Governance Invariants (AFOS v30.1 & V3 Kernel)

### 1.1 Environmental Bifurcation (Path A vs. Path B)
* **Path A (`core/`, `adam_*/`, `src/`, `afos_v3/`, `skills/`):** Strict Pydantic/dataclass typing, defensive error handling, deterministic math via Rust/Python kernels (`jsonLogic`), and mandatory W3C PROV-O `ProvenanceHeader` telemetry.
* **Path B (`experimental/`, `research/`, `tinker_lab/`):** Expansive velocity and rapid prototyping. **NEVER** import Path B modules into Path A execution paths.
* **UI Presentation Isolation:** Never import UI libraries (`streamlit`, `react`, `dash`, `gradio`) inside `core/`, `adam_*/`, or `src/`. All presentation logic stays strictly in `frontend/`, `services/webapp/`, or `showcase/`.

### 1.2 AFOS V3 Zero-Trust Ingestion & 75/25 Barbell Circuit Breaker
* **mTLS & Ephemeral STS Credentials (`src/api/routers/v3_ingest.py`, `afos_v3/gateway/zero_trust_gateway.py`, `src/ingestion/zero_trust/`, `src/zero_trust/`):**
  * All ingestion nodes (Whalescanner, Macro Telemetry) must authenticate via TLSv1.3 mTLS (`CN=AFOS-V3-ROOT-CA`, `ECDSA-p256`) and short-lived ECDSA-signed `EphemeralSTSToken` credentials with a hard maximum $900\text{-second}$ ($15\text{-minute}$) TTL, alongside `BoundedMarketMetrics` physical bounds validation.
* **W3C PROV-O Bit-Level Checksum Verification:**
  * Every ingested payload is canonicalized (`json.dumps(payload, sort_keys=True)`) and verified against `expected_sha256_hash`.
  * A single-bit mismatch immediately diverts the payload to `QuarantineSink` (`global_quarantine_matrix`) with forensic tags `["BIT_MISMATCH", "PROV_O_INTEGRITY_VIOLATION"]` and triggers the system circuit breaker.
* **Triangulated 3-Feed Consensus & $0.5\sigma$ Outlier Severing (`afos_v3/consensus/triangulated_engine.py`, `src/zero_trust/consensus.py`):**
  * Core macro telemetry (`sp500_price`, `us10y_yield`, `vix`, `btc_usd`, `brent_crude`) is fetched concurrently across $3$ independent Tier-1 feeds.
  * Any feed deviating by $> 0.5\sigma$ (above the $10\text{ bps}$ microstructure noise floor) is severed and logged as `Adversarial_Spoofing_Attempt`. A minimum quorum of $2\text{ of }3$ verified feeds is required.
* **75% Fortress / 25% Convex Hunt LOCKDOWN State Machine (`src/adam/core/state_manager.py`, `briefings/fortress_hunt_20261027.md`):**
  * **Nominal State (`NORMAL` / `NOMINAL`):** Routes $75\%$ of NAV to Fortress Ballast ($50\%$ Sub-90D US T-Bills + $25\%$ AAA Senior CLO Tranches at $\text{SOFR}+135\text{ bps}$) and $25\%$ to Convex Hunt (asymmetric alpha/catalyst execution, e.g., Long coastal refiners VLO capturing Brent-WTI crack spreads / Short levered BSL maturity-wall targets ATUS with $\text{DSCR} < 0.75\times$, paired with a 60D SPX zero-cost put spread collar).
  * **Lockdown State (`LOCKDOWN`):** On consensus failure, bit mismatch, spoofing detection, or physics breach, `SYSTEM_STATUS` transitions to `LOCKDOWN`, immediately halting the $25\%$ Convex Hunt execution leg (`convex_hunt: 0.00`) while preserving the $75\%$ Fortress allocation. Resetting `LOCKDOWN` to `NORMAL` requires verified meatspace cryptographic authorization (`MEATSPACE_AUTH_APPROVED_V3`).

### 1.3 Inter-Agent Routing, Boundary Protocols & Append-Only Telemetry
* **No Direct Agent Instantiation:** AgentA must never instantiate AgentB directly to prevent circular dependencies. Cross-domain requests must be routed through `MetaOrchestrator` / `OmegaMetaOrchestrator` (`core/agents/meta_agents/omega_meta_orchestrator.py`) or Nexus via output metadata:
  ```python
  metadata={"next_step": "invoke_surveillance", "query": "verify_covenant_compliance"}
  ```
* **System $1 \rightarrow \text{System } 2$ Boundary Protocols:** Across System 1 to System 2 transitions, enforce `SwarmToGraphBoundary` (`SwarmOutput -> GraphInput` in `adam_interfaces/protocols.py`) and `FastMCPToolMapper` (`adam_swarm/orchestrator.py`).
* **Append-Only Journals ("Flash-Memory Amnesia" Prevention):** Never use `open(f, 'w')` on logs or agent journals (`.jules/history.jsonl`, `SwarmLogger`). Always read context first and use `open(f, 'a')` or bitemporal append-only ledgers (`valid_at`, `transaction_at`, `sha256`).

---

## 2. Sentinel's Watch: Security & Zero-Trust Hardening (2024–2026 Full Ledger)

### 2.1 P0 Critical Vulnerability Matrix

| Vulnerability Class | Historical Finding & Root Cause | Mandatory Prevention Rule |
| :--- | :--- | :--- |
| **Insecure Deserialization (`pickle`)** | `pickle.load()` / `pickle.loads()` in `core/analysis/technical_analysis.py` (`TechnicalAnalystAgent`, sometimes masked with `# nosec B301`) and `src/adam/core/state_manager.py` (`StateManager` Redis payloads) allowed Arbitrary Code Execution (RCE). | **NEVER** use `pickle.load()`, `pickle.loads()`, or `# nosec B301` bypasses. Always use `core.security.safe_unpickler.safe_load()` / `safe_loads()` (whitelisting numpy, pandas, torch) or safer formats (JSON, ONNX, Safetensors, skops). |
| **Dynamic Import RCE & File Shadowing** | Endpoints like `/api/simulations/<name>` using `importlib.import_module(name)` allowed RCE on module import side-effects. Legacy files like `core/api.py` shadowed by packages (`core/api/`) remained executable and leaked stack traces. | **NEVER** pass user input directly to `import_module`; enforce a strict static dictionary mapping or filesystem allowlist. Audit and remove legacy `.py` files shadowed by package directories of the same name. |
| **Dynamic Code Execution (`eval` / `exec` / `str.format`)** | `exec()` in `server/server.py` and `server/mcp_server.py` (`execute_python_sandbox`) and `eval()` in `framer.py` enabled RCE. In `SecureSandbox` (`core/security/sandbox.py`), `"{0.__class__.__globals__}".format(obj)` bypassed AST private-attribute checks via C-level format evaluation. | **NEVER** use `eval()` or `exec()`. Walk `ast.parse()` against an explicit node whitelist (`_safe_eval_condition`). Inside `SecureSandbox`, ban `format` and `format_map` alongside AST validation, restricted globals, process isolation, execution timeouts, and `GovernanceEnforcer` conviction scoring (`core/security/governance.py`). |
| **Command & Option Injection (`git`, `bash`)** | `GitRepoSubAgent` and `GitHubAlphaAgent` ran `subprocess.run(["git", "clone", repo_url, repo_path])` and `WorkspaceManager._run_hook` ran `["bash", "-lc", script]`. Payloads starting with `-` (`--upload-pack=...`, `-i`) or `ext::` / `file://` schemes triggered RCE. | 1. Validate URL schemes strictly (`https://` only; block `ext::` and `file://`).<br>2. Always insert the end-of-options separator `--` before positional arguments (`["git", "clone", "--", repo_url, repo_path]` and `["bash", "-lc", "--", script]`). |
| **SSRF via HTTP Redirects & Private IPs** | `SupplyChainRiskAgent` and `SecurityGovernanceGatekeeper` (`src/pdil/middleware.py`, `src/pdil/network_validator.py`) validated initial URL domains against a whitelist, but `urllib.request.urlopen` followed HTTP 30x redirects to `127.0.0.1` or cloud metadata IPs. | Enforce `https://` only, resolve DNS via `socket.gethostbyname` and block private/loopback IPs (`ipaddress.ip_address(ip).is_private` or `is_loopback`), enforce `ssl.create_default_context()` (never `ssl._create_unverified_context()`), and disable automatic redirects via a custom `NoRedirectHandler(urllib.request.HTTPRedirectHandler)` returning `None` for `redirect_request`. |
| **Path Traversal in File & MCP Endpoints** | File endpoints (`CodeAlchemist.deploy_to_local_file`, `scripts/server.py` `/api/run_rag`, MCP `ingest_file` in `server/mcp_server.py`, `GitRepoSubAgent`) accepted unvalidated paths (`../../secret.env`). | Resolve with `os.path.realpath` / `os.path.abspath` and verify `os.path.commonpath([resolved, SAFE_ROOT]) == SAFE_ROOT` (with trailing `os.sep` check, restricted to `data/` and `docs/`). |
| **XML External Entity (XXE)** | `news_bot.py` and XBRL handlers parsed untrusted RSS/SEC XML using `xml.etree.ElementTree`. | Always use `defusedxml` for all XML/XBRL/RSS parsing and declare `defusedxml` in `pyproject.toml` and `setup.py`. |
| **SQL Injection via Keyword Blacklists** | `LakehouseConnector` (`core/data_access/lakehouse_connector.py`) and LLM SQL tools checked if `"DROP"` in query or `startswith("SELECT")`, failing to block `UNION SELECT` secrets, `INSERT`, `ALTER`, `TRUNCATE`, `ATTACH`, or stacked queries. | Never construct SQL with f-strings or rely on keyword blacklists. Use parameterized ORM queries, a tokenizer-based single-statement `SELECT` validator, and `sqlite3.set_authorizer` table/action whitelists with read-only URI mode (`?mode=ro`). |
| **Log Leakage & Unauthenticated Endpoints** | `/api/state` exposed raw server logs containing secrets; `/api/agents/<agent_name>/invoke` and `core/api.py` (`/`) lacked authentication; `/api/generate_credit_memo` used `@jwt_required(optional=True)` enabling compute DoS. | **NEVER** expose internal log buffers via public APIs. Enforce `X-API-Key` or JWT middleware on all command/orchestration endpoints. For showcase endpoints with optional JWT, use Adaptive Security: return a lightweight static/simulation fallback for unauthenticated callers instead of running heavy compute. |

### 2.2 High-Severity P1 Security Mandates
* **Hardcoded & Weak Cryptographic Secret Defaults:**
  * **Finding:** Fallbacks like `os.environ.get("SECRET_KEY") or "default"`, Neo4j `"password"`, `streamlit_portal.py` `admin/admin`, `YOUR_NEWS_API_KEY` in `event_driven_risk_agent.py`, and `"default-insecure-key-change-me"` in `core/settings.py`.
  * **Directive:** Never commit placeholder secret strings or static fallbacks. Fail fast (`RuntimeError` on startup in non-dev environments) or use `Field(default_factory=lambda: secrets.token_urlsafe(32))` for ephemeral runtime keys.
* **Web Framework, WebSocket (CSWSH), & Template Hardening:**
  * **CORS & Security Headers:** Configure explicit `CORS(resources={...})` origins and inject CSP (`default-src 'self'`), HSTS, `X-Frame-Options: DENY`, and `X-Content-Type-Options: nosniff` in `after_request`.
  * **WebSocket Origin Validation:** FastAPI `CORSMiddleware` only covers HTTP, not WebSockets. Explicitly validate the `Origin` header inside `/ws/mesh` and `/ws/stream` handlers against an allowed origins list.
  * **Interface Binding & SSL Contexts:** Bind local Uvicorn/FastAPI services (`services/sentinel_api.py`, `server/sentinel_api.py`) to `127.0.0.1` rather than `0.0.0.0`. Always enforce certificate verification via `ssl.create_default_context()`.
  * **Jinja2 Autoescaping (`build_dashboard.py`):** Always instantiate `Environment(..., autoescape=True)` (or `select_autoescape`, Bandit B701) to prevent XSS when rendering HTML artifacts, and validate user registration/profile inputs against strict alphanumeric allowlists.
* **Supply Chain, Timeouts & Fail-Closed Governance:**
  * Pin Hugging Face `from_pretrained(..., revision="<40-char-commit-sha>")` calls (Bandit B615).
  * Always pass an explicit `timeout=` parameter to every `requests.get`/`post` and `urllib.request.urlopen` call (e.g., in `EventDrivenRiskAgent` and `downloads/download_agents.py`) to prevent thread starvation (CWE-400).
  * Ensure `GovernanceMiddleware` fails closed (`abort(403)`) on HIGH/CRITICAL risk classifications rather than logging with `pass` ("Security Theater").

---

## 3. Bolt's Journal: Architecture, Performance, Vectorization & Concurrency

### 3.1 Structural Traps (P1)
* **Duplicate Knowledge Graph Classes (ACTIVE RISK):**
  * **Finding:** `core/engine/unified_knowledge_graph.py` and `core/v23_graph_engine/unified_knowledge_graph.py` are distinct but similar (`NeuroSymbolicPlanner` imports `core/engine`). Historical PoCs were moved to `archive/v23_prototypes/` (superseded by `core/engine/cyclical_reasoning_graph.py`).
  * **Directive:** Always verify imports. Prefer `core/engine/unified_knowledge_graph.py` as canonical. Do not edit one without checking the other.
* **Duplicate Scrubber & Deterministic Hash Logic:**
  * **Finding:** `core/data_processing/utils.py` and `core/data_processing/universal_ingestor.py` duplicated text cleaning logic (`GoldStandardScrubber`); `src/schemas/core_types.py` contained duplicate `compute_deterministic_hash` definitions.
  * **Directive:** Consolidate shared text-cleaning utilities in `core/data_processing/utils.py` / `core/common/text_utils.py` with pre-compiled class-level regexes (`re.compile(..., re.IGNORECASE)`) and maintain a single canonical `compute_deterministic_hash` in `src/schemas/core_types.py`.

### 3.2 Python, Pandas & Numerical Kernels (P2)
* **Pre-Compile jsonschema Validators ($50\text{–}100\times$ Speedup):**
  * Never call `jsonschema.validate(instance, schema)` inside a loop (`scripts/eval_001a_verifier.py`, `GovernanceGatekeeper`).
  * Instantiate once via `ValidatorClass = jsonschema.validators.validator_for(schema); validator = ValidatorClass(schema)` and reuse `validator.validate(instance)`.
* **Eliminate `df.iterrows()` Across All Pandas Pipelines ($4\text{–}100\times$ Speedup):**
  * Never use `df.iterrows()` in `DataFetcher.fetch_historical_data`, `WhaleScanner.calculate_fund_sentiment`, `YFinanceMarketData`, `InstitutionalRadarAnalytics.detect_cluster_buys`, or `AgentImprovementPipeline`.
  * For dict serialization: use vectorized column operations + `df.reset_index()[cols].to_dict(orient="records")` (or `dropna() + to_dict("index")` for nested MultiIndex frames).
  * For lightweight iteration without dict allocation: use `df.itertuples()`.
  * For multi-column tuple set membership (`(fund_name, cusip)`): use `mask = ~curr_df.set_index(["fund_name", "cusip"]).index.isin(prev_holdings)`.
* **Vectorized Monte Carlo, Quantum Batching & Multi-Ticker Threading:**
  * Replace Python simulation loops in Geometric Brownian Motion (2.9s $\rightarrow$ 0.36s, $8\times$) and `QuantumMonteCarloEngine` (0.35s $\rightarrow$ 0.076s for $100\text{k}$ sims, $4.6\times$) with vectorized NumPy arrays (`np.random.normal`, `np.cumsum`, `np.exp`) and batched circuit execution (`SimulatorBackend.run_circuit_batch()`).
  * Use threaded `yf.download(tickers, group_by="ticker")` instead of sequential `yf.Ticker(sym).history()` calls (1.49s $\rightarrow$ 0.33s, $4.5\times$).
* **Combinatorial Order-Book Hoisting (`UniversalArbitrageEngine`):**
  * In $O(E^2)$ cross-exchange arbitrage loops, hoist $O(L)$ order book depth walks (`_walk_book`) and `_calculate_micro_price` outside the nested $(i, j)$ loops into precomputed $O(1)$ lookup dictionaries, and reuse bounded volume cost/revenue when $\min(\text{exec\_vol\_buy\_a}, \text{exec\_vol\_sell\_b}) == \text{target\_volume}$.

### 3.3 Caching, Graph Batching, CQRS Indexing & Async CPU Offloading (P2)
* **Singleton & Class-Level Config/Graph Caching:**
  * Cache `UnifiedKnowledgeGraph`, `jsonLogic` precondition loaders (`AgentOrchestrator._evaluate_preconditions`), and YAML config readers (`load_config`) using `@functools.lru_cache` or class-level caches (`DataLoader._CLASS_WHITELISTED_APIS`). Always call `cache_clear()` in test fixtures to prevent cross-test pollution.
  * Layer an in-memory `OrderedDict` LRU cache ($1,000$ items) in front of disk-backed `SemanticCache` lookups.
  * Initialize Neo4j and Redis drivers once at application startup (singleton), never inside per-request handlers.
* **NetworkX Batch Ingestion ($\sim 30\%$ Speedup):**
  * Batch node and edge insertions into lists and call `graph.add_nodes_from()` / `graph.add_edges_from()` rather than calling `add_node()` inside loops (especially on $100\text{k}+$ nodes and in `ingest_risk_state`).
* **$O(1)$ CQRS Event Ledger Indexing:**
  * Maintain `_aggregate_index[aggregate_id]` and `_correlation_index[correlation_id]` hash maps on event insertion rather than scanning `self._events` with $O(N)$ list comprehensions during replay.
* **Offload CPU Work in Async Pipelines:**
  * Using `aiofiles` in `UniversalIngestor.scan_directory_async` is insufficient if `json.loads`, regex scrubbing, `ast.parse`, or `sentence_transformers` embedding generation runs on the main event loop. Wrap CPU-bound steps in `await asyncio.to_thread(...)`.

### 3.4 Frontend & React Streaming Dashboard Optimization (P2)
* **Zustand Selector Isolation & Deterministic Jitter:**
  * Never destructure the full Zustand store (`const { agents } = useSwarmStore()`) in parent views like `PromptAlpha.tsx` when simulations tick at $5\text{ Hz}$ (causes full $O(N)$ page re-renders). Subscribe only to invariant actions via selectors (`useSwarmStore(s => s.initializeSwarm)`) and read volatile state inside loops via `useSwarmStore.getState()`.
  * Never call `Math.random()` inside `useMemo`. Push visual jitter down to `React.memo` leaf components using a deterministic PRNG seeded by `id + tick`.
* **Decoupled Input State, `React.memo` & Stable UUID Keys on Sliding Windows:**
  * Separate `InputState` from `HistoryState` so terminal history does not re-render on every keystroke; debounce range sliders in `ScenarioSimulator` with `useEffect + useRef`.
  * Extract mapped row/card elements (`AgentCell`, `ThoughtRow`, `AgentIntercom`, `NewsWire.tsx`, terminal lines) into `React.memo` components.
  * Never use array index (`key={i}`) or content substrings on sliding-window lists (`prev.slice(-50)`), as prepending/slicing shifts every index and forces full $O(N)$ re-renders. Assign a stable UUID at item creation time, pass explicit boolean props (`isNewest`) for CSS flash animations, and move `<style>` / `@keyframes` definitions to global `index.css`.
* **Search Indexing, Sorting, Filtering & Recursive Polling:**
  * **Fuse.js Split useMemo (`GlobalNav.tsx`):** Use one `useMemo` to instantiate `new Fuse(manifest)` (depending only on `manifest`), and a second `useMemo` to run `fuse.search(searchTerm)` on keystrokes.
  * **Sorting & Filtering (`PromptAlpha.tsx`, `HFTView`):** Wrap list sorts/filters in `useMemo`; subtract numeric timestamps directly (`a.timestamp - b.timestamp`) instead of allocating new `Date().getTime()`; hoist `filter.toLowerCase()` outside `.filter()` callbacks with an early return when `!filter.trim()`.
  * **Recursive `setTimeout` Polling (`NarrativeDashboard`, `AuditLogViewer`, `usePromptFeed.ts`):** For async network polling, replace `setInterval` with recursive `setTimeout` wrapped in `try/catch` (so transient network errors never kill the polling loop), guarded by `isMounted`, and skip `setState` when `JSON.stringify(prev) === JSON.stringify(data)`. (Keep standard `setInterval` only for synchronous local mock generators like `NewsWire.tsx`).
* **DOM TreeWalker Early Exit (`showcase/js/dashboard-logic.js`):** Terminate `TreeWalker` traversal immediately once the maximum match count ($20$) is reached (58ms $\rightarrow$ 0.3ms).

---

## 4. Palette's Standards: UX, Accessibility (WCAG/ARIA) & Cyberpunk UI

### 4.1 Terminal & Streaming Log Regions
* Annotate dynamic terminal outputs with `role="log"` and `aria-live="polite"`.
* Support `ArrowUp` / `ArrowDown` command history navigation in terminal inputs.

### 4.2 Keyboard-Scrollable Containers
* Any container with `overflow: auto` (`overflow-y-auto`) must include `tabIndex={0}`, `role="region"`, `aria-label`, and visible focus rings (`focus:ring`).

### 4.3 Interactive Semantics & Roving TabIndex
* **Buttons vs. Divs:** Never bind `onClick` to a bare `<div>` for navigation, mode switchers, or news feed items; always use semantic `<button>` elements.
* **Tabs:** Implement `role="tablist"` and `role="tab"` with `ArrowLeft` / `ArrowRight` / `Home` / `End` roving `tabIndex`.
* **Custom Dropdowns:** Implement `role="combobox"` with `aria-activedescendant` and `ArrowUp` / `ArrowDown` keyboard handlers.
* **Global Search:** Pair `Ctrl+K` / `Cmd+K` shortcuts with a magnifying glass icon, `aria-label`, and a right-aligned `<kbd>` badge (extracted out of placeholder text).

### 4.4 Custom Progress Bars, Telemetry Meters & Forms
* Annotate custom `<div>` progress/telemetry meters with `role="progressbar"` (or `role="meter"`), `aria-valuenow`, `aria-valuemin`, `aria-valuemax`, and link to visual labels via `aria-labelledby`.
* Generate deterministic `id` / `htmlFor` pairs for all schema-generated and legacy form inputs.

### 4.5 Loading, Empty, Raw Editor States & Component Tree Verification
* **Spinners & Action Buttons:** Pair spinners with `role="status"` and visually hidden "Loading..." text; swap button icons for a spinner and update label text ("Analyzing...") while disabled during execution.
* **Empty States:** Render a dedicated `role="status"` empty state whenever dashboard arrays are empty.
* **Raw Data Editors:** Bind JSON editors to a local draft string state and only parse/commit to the object store on valid blur or submit.
* **Component Tree Verification:** Always trace `App.tsx` / `Layout.tsx` to verify which component (`GlobalNav.tsx` vs. legacy `Header.js`) is actually mounted before refactoring, and preserve exact cyberpunk neon hex tokens (`text-[#00f3ff]`) in Tailwind migrations.

---

## 5. Swarm Intelligence, PDIL, Harness 2 & Kernel Architecture (ADAM-IAK-CORE)

### 5.1 ADAM Intent & Attention Kernel (ADAM-IAK-CORE Spec 1.0.0)
Implemented in `src/iak/adam_iak_primitives.py` and `adam_iak_primitives.py`:
* **`DtypeSafeMaskedEntmaxSoftmax` ($1.5\text{-Entmax}$ Sparse Attention):**
  * Upcasts low-precision inputs (`fp16` / `bf16`) to `fp32` prior to root-finding bisection (`bisect_threshold_entmax15`, $p_i = [\frac{1}{2}(z_i - \tau)]_+^2$ with exact lower bound $\tau_{\min} = z_{\max} - 2.0$).
  * Applies $1/\sqrt{d_k}$ logit scale normalization using `torch.std(..., correction=0)` guarded by `scores.size(-1) > 1` (preventing single-element NaN).
  * Applies mask fills before threshold computation, and explicitly zeros out fully-masked rows to prevent uniform NaN/leakage across padding tokens.
* **`UncertaintyGatedCRFSlotDecoder` (Helmholtz Energy OOD Gate + Viterbi BIO):**
  * Computes Helmholtz free energy $E(x) = -T \ln \sum_k \exp(z_k / T)$ across token emissions.
  * If sequence uncertainty exceeds the calibrated epistemic threshold, the decoder gates the sequence to the `NULL_INTENT` baseline and enforces valid B- / I- / O transitions via linear-chain Viterbi decoding returning `(best_paths, best_scores)`.
* **`DeterministicPolicyEpistemicGate` (6-Tier Epistemic Lattice):**
  * Enforces a strict 6-tier governance lattice (`VERIFIED`, `CALCULATED`, `MODEL_EST`, `HYPOTHETICAL`, `THIN`, `REJECTED`) with fail-closed zero-default financial rejection when ungrounded arithmetic or policy breaches occur.

### 5.2 W3C PROV-O, PDIL Gatekeepers, HNASP & Dual-Model Arbitration
* **`ProvenanceHeader` (`src/pdil/models.py`, `docs/architecture/PDIL_MAPPING.md`):** Requires `git_commit_hash`, `timestamp`, `content_hash` (SHA-256 of canonical `json.dumps(payload, sort_keys=True)`), `jsonLogic_version`, `confidence_score`, `derivation_path`, and `source_data_object`. `hash_state()` hashes `git_commit_hash:jsonLogic_version`.
* **`AgentOutput.check_grounding()` (`src/schemas/core_types.py`, `core/schemas/agent_schema.py`):** Verifies non-empty `source_data_object` grounding; tested via hypothesis property tests in `tests/test_pdil_property.py` and `ops/checks/check_provenance.py` (`variable_node` entries must include `context_provenance`).
* **PDIL Middleware (`src/pdil/middleware.py`, `src/pdil/network_validator.py`):** Houses `SecurityGovernanceGatekeeper`, `JsonLogicGovernanceGatekeeper`, `DriftIntelligenceLayer`, `ProofOfThoughtLogger`, and `MilestoneLogger` (`adam_governance/state_control.py`).
* **`ModelArbitrationOutput` (`afos/afos/core_types.py`):** Enforces `ConfigDict(strict=True, extra="forbid")` with `bidirectional_disparity`, `arbitration_flag_triggered`, $\text{one\_sided\_downside\_spread} = \max(0, \text{PD}_\beta - \text{PD}_\alpha)$, and `capital_buffer_penalty`.
* **`HNASPState` (`core/schemas/hnasp.py`):** Tracks `Meta`, `PersonaState` (`EPAVector`), `LogicLayer` (evaluated via `jsonLogic` before `_original_execute`), `ContextStream`, and `ExecutionTrace`.

### 5.3 ADAM Harness 2 Reflective Evolution (`adam_harness2`, `tests/test_adam_harness2.py`)
* **`HarnessConfig`:** Computes a deterministic 12-character `fingerprint()`. Component `guardrails` is immutable (raises `PermissionError` if a patch attempts to weaken guardrails); `plugins` is strictly append-only.
* **`Harness2Controller` & `ExperienceLibrary`:** Runs the 3-stage Editor $\rightarrow$ Composer $\rightarrow$ Reflector loop (`prompts/harness2_stage_prompts.py`) over `ExecutionTrace` probes, committing verified domain lessons to `ExperienceLibrary` and streaming events to append-only JSONL telemetry.

---

## 6. Quantitative Credit Forensics, Market Contagion & Multi-Horizon Memory

### 6.1 Off-Balance-Sheet Compute Lease Capitalization (SR 11-7 / SR 26-2 / ASC 842 Forensics)
* **Model Risk Modernization (SR 26-2 / OCC 2026-13):** On April 17, 2026, the Federal Reserve, OCC, and FDIC jointly issued SR 26-2 (superseding SR 11-7 and SR 21-8) establishing a proportionality framework for institutions with >$30B assets and model materiality. While traditional quantitative models (TimesFM quantiles, Sobol QMC, CRE 31 PD) are validated under SR 26-2, agentic swarms are governed by real-time technology risk, kill-switches, and the **W3C PROV-O Root-Source Disjointness Engine** (`afos_v3/prov_o_consensus.py`, enforcing pairwise Jaccard source overlap $\le 0.20$ across $K \ge 3$ distinct institutional modalities) and the **Neuro-Symbolic Risk Adapter Gate** (`adam_finance/risk_adapter_gate.py`, enforcing Multiplicative Assurance $\text{Layer 0} \times \text{Layer 1}$ and lexicographic safety tripwires).
* **Finding:** Frontier AI issuers (e.g., Anthropic Form S-1 forensic coverage in `showcase/data/adam_daily/2026-10-26/daily_brief.md` and `reports/anthropic_s1_credit_briefing_oct2026.md`) may report $\$0.0\text{M}$ in traditional GAAP funded debt while carrying $\$224.5\text{B–}\$430.0\text{B}$ in non-cancelable off-balance-sheet cloud compute and equipment commitments in footnotes, alongside GAAP net losses dominated ($89.7\%$) by non-cash convertible note fair-value mark-to-market charges.
* **Directive:** Never underwrite AI or cloud-intensive obligors on GAAP funded debt alone ($0.0\times$ leverage illusion). Strip non-cash convertible derivative marks to isolate Operating Cash Flow, capitalize non-cancelable take-or-pay compute minimums at the implicit lease discount rate ($6.0\%$) into a Lease-Capitalized Credit Basis ($\text{Debt}_{\text{Funded}} / \text{EBITDAR}$), and model Full Cash-Flow Waterfall CADS:
  $$\text{CADS}_{\text{Full Waterfall}} = \text{EBITDAR} - \text{CapEx}_{\text{Maint}} - \text{Taxes}_{\text{Cash (IRC 163(j))}} - \Delta\text{WC}_{\text{Leases}}$$
  Include drawn RCF interest and undrawn RCF commitment fees ($37.5\text{ bps}$) across both Flat Spot SOFR and Stagflationary Hike-Path stress regimes.

### 6.2 Hyperscaler Circular Credit Loop & GPU SPV Contagion
* Track the transmission channel between hyperscaler IG debt issuance ($\$175\text{B+}$ YTD funding $\$650\text{B–}\$785\text{B}$ AI CapEx) and variable, consumption-based per-token enterprise revenue at frontier AI tenants. A $\ge 35\%$ annualized token price deflation without $> 1.6\times$ volume elasticity is a primary Fixed-Charge Coverage Ratio ($\text{FCCR} < 1.15\times$) tripwire.

### 6.3 Liability Management Transactions (LMT) & Covenant Blockers (`scripts/lmt_contagion_gate_orchestrator.py`)
* Always audit Credit Agreement Section 7.02 negative covenants for J.Crew / Chewy / Envision IP drop-down blockers (preventing model weights or core IP from being transferred to unrestricted subsidiaries) and Serta non-pro-rata uptiering blockers.
* Enforce strict Absolute Priority Rule (APR) senior-to-junior recovery monotonicity across restructuring trees.

### 6.4 Credit Dominance Rules (`MarketSentimentAgent.check_credit_dominance_rule`)
* Validated in `tests/test_adam_v24_logic.py`:
  * Trigger "Liquidity Mirage" when $\text{spy\_change\_pct} > 0.5\%$ while $\text{relative\_credit\_performance} < -0.15\%$.
  * Trigger "Systemic Tremor" whenever the Treasury curve inverts (`^TNX < ^IRX`).

### 6.5 T+1 Settlement Shift & Multi-Horizon Continuum
* Apply a $+5\text{ to }+25\text{ bps}$ baseline illiquidity premium modifier for T+1 settlement friction and localized intraday repo funding spikes around 15:00 EST.
* Explicitly map quantitative stress tests, simulations, and liquidity models across all $5$ operational timebands: Real-Time/High Speed, Intraday, Daily/Weekly, Monthly/Quarterly, and Long Term ($1, 5, 10, 30\text{ Year}$).

### 6.6 Centralized Artifact Indexing & Pricing Database Dashboard
* Register all legacy and newly generated daily briefings and `showcase/*.html` visualizers in `showcase/adam_daily_hub_index.html` and map them to a designated "Engine Horizon" in `pricing_intelligence_hub.html`.
* Extract the core security, numerical pricing impact, confidence $\%$, and categorical conviction level from every structural report and append them to `pricing_database_dashboard.html`.
* Validate JSONL telemetry with `ops/checks/check_provenance.py` (`variable_node` requires `context_provenance`).

---

## 7. Jules CLI (@google/jules), Local SDK & Autonomous Workflow Memory

### 7.1 Minimal-Privilege Credential Pass-Through (`scripts/export_jules_data.py`)
* Never invoke `security find-generic-password` (macOS Keychain) or `secret-tool` from Python automation scripts. Delegate authentication directly to the `@google/jules` CLI binary (`jules remote list --repo`, `jules remote list --session`, `jules remote pull --session <id>`) or pass ephemeral tokens strictly in memory via `JULES_ACCESS_TOKEN`.

### 7.2 Architect Agent (`core/agents/architect_agent.py` & `scripts/run_jules_workflows.py`)
* Use `JulesCommandBuilder` and `ArchitectAgent` to validate repository targets (`SAFE_REPO_PATTERN`), prevent shell/argument injection (`shlex.quote`), audit Path A code snippets for UI/Path B/P0 violations, and attach SHA-256 W3C PROV-O `ProvenanceHeader` traces to every dispatched coding session.

### 7.3 Local-First SDK & Web IDE (`src/jules/`)
* Comprises `JulesAgent`, `JulesEngine`, `JulesConfig`, `JulesMemoryManager` (`.jules/memory.json`, append-only `.jules/history.jsonl`), `JulesToolRegistry` (recording `urn:jules:tool_exec:<tool>:<timestamp>` with `context_provenance`), and the Web IDE (`src/jules/ide/server.py`, `jules_ide.html`).

---

## 8. Prioritized Engineering Backlog

> **Task for Agents:** Select items from this list based on Priority (P0 = Highest).

### P0: Security & Zero-Trust Enforcement (Sentinel)
- [x] **AFOS V3 Zero-Trust Ingestion Gateway:** Implemented mTLS (TLSv1.3), $15\text{-min}$ ECDSA STS tokens, W3C PROV-O SHA-256 bit-mismatch quarantine, and 3-feed $0.5\sigma$ consensus with 75/25 Barbell LOCKDOWN circuit breaker.
- [x] **Audit `importlib` & Secure `pickle`:** Replaced `pickle.load` / `pickle.loads` with `core.security.safe_unpickler` and restricted dynamic imports.
- [x] **Fix SQL Injection:** Replaced `f"SELECT ... {var}"` with parameterized queries, tokenizer SELECT validation, and `sqlite3.set_authorizer`.
- [x] **SSRF Redirect & Jinja2 XSS Prevention:** Disabled automatic redirects in `SecurityGovernanceGatekeeper` and enabled `autoescape=True` in `build_dashboard.py`.
- [x] **API Auth & Adaptive JWT Fallback:** Enforced `X-API-Key` and Adaptive JWT fallback across command endpoints.
- [ ] **Continuous `importlib` & `pickle` CI Gate:** Enforce automated CI gate blocking any new `pickle.load` or unwhitelisted `importlib.import_module` across `core/` and `src/`.

### P1: Kernel, Architecture & Performance (Bolt)
- [x] **ADAM-IAK-CORE Primitives (`src/iak/`):** Remediated `DtypeSafeMaskedEntmaxSoftmax`, `UncertaintyGatedCRFSlotDecoder`, and `DeterministicPolicyEpistemicGate`.
- [x] **Pre-Compiled JSONSchema & Vectorized DataFrames:** Eliminated `df.iterrows()` in `WhaleScanner`/`DataFetcher` and pre-compiled `jsonschema` validators in `eval_001a_verifier.py`.
- [x] **Deduplicate Scrubbers & Archive Graph PoCs:** Unified `GoldStandardArtifact` and `GoldStandardScrubber` in `core/data_processing/utils.py`; moved `experimental/v23_prototypes/v23_graph_engine` to `archive/v23_prototypes/`.
- [x] **Multi-State Reasoning Grader:** Added `evals/graders/multi_state_reasoning_evaluator.py` evaluating trace fidelity, fact insertion $F_1$, reasoning rubrics, and ECE.
- [x] **Fix Async & Append-Only Loggers:** Ensured all Swarm and Jules agents use append mode (`open(..., 'a')`).
- [ ] **Consolidate Legacy Graph Imports:** Complete migration of remaining `core/v23_graph_engine` references into `core/engine/unified_knowledge_graph.py`.
- [ ] **Complete Scrubber Consolidation:** Merge any remaining duplicate text-cleaning logic in `universal_ingestor.py` into `core/data_processing/utils.py`.

### P2: UX & Accessibility (Palette)
- [x] **Debounce Inputs & Split Fuse.js Memoization:** Debounced simulation range sliders and separated Fuse.js index instantiation from search execution.
- [x] **Daily Hub Index Synchronization:** Updated `showcase/adam_daily_hub_index.html` and provenance checks in `ops/checks/check_provenance.py`.
- [ ] **ARIA Live & Roving TabIndex Audit:** Run a full accessibility audit across `services/webapp/client` to verify `role="log"`, `aria-live="polite"`, keyboard roving `tabIndex`, and stable UUID keys across all streaming widgets.

### P3: Documentation & Governance Housekeeping
- [x] **Update Tutorials & PDIL Mapping:** Added `docs/architecture/PDIL_MAPPING.md`, `docs/jules_tools_reference.md`, and `docs/how-to/using_jules_cli_ide_sdk.md`.
