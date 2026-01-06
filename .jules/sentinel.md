## 2025-12-10 - Flask Error Leakage & Broken Tests
**Vulnerability:** Flask's default error handler leaks exception details (including SQL queries and table names) to the client. Additionally, unit tests were mocking the wrong method (`run_agent` instead of `execute_agent`), causing tests to pass even when the underlying code was broken (returning 500s).
**Learning:** Default Flask configuration is not secure for production. Mocks in tests can mask broken code if they don't match the actual implementation.
**Prevention:** Always implement a custom error handler that returns generic messages. Verify mocks against the actual class interface (e.g., using `autospec=True` or careful review).

## 2025-12-11 - Permissive CORS Configuration
**Vulnerability:** The Flask API was configured with `CORS(app)` without arguments, which defaults to allowing all origins (`*`) and reflecting the origin header. This exposes the API to Cross-Site Request Forgery (CSRF) and data exfiltration from malicious sites.
**Learning:** `flask-cors` is permissive by default ("Allow All") to simplify development, but this is dangerous if not explicitly restricted.
**Prevention:** Always configure `resources` with a specific list of `origins` in `CORS()`, preferably loaded from environment variables to allow flexibility across environments (dev vs prod).

## 2024-05-23 - API Log Leakage
**Vulnerability:** The `/api/state` endpoint exposed the last 50 raw server log entries to unauthenticated users. This could leak sensitive data like API keys, PII, or internal system paths logged at INFO level.
**Learning:** In-memory log buffers exposed via API are a high-risk pattern. Developers often add them for "easy debugging" in UI, forgetting that logs often contain secrets.
**Prevention:** Never expose raw server logs via public APIs. Use proper observability tools (ELK, Splunk, CloudWatch) with access controls. If UI needs logs, stream them via authenticated WebSockets and sanitize them.

## 2025-12-14 - Dynamic Import RCE Risk
**Vulnerability:** The `/api/simulations/<name>` endpoint passed the `name` parameter directly to `importlib.import_module()`, allowing potential Arbitrary Code Execution if a user could invoke a module with side-effects on import.
**Learning:** Dynamic imports based on user input are dangerous. Relying on "it's just an import" is insufficient security.
**Prevention:** Whitelist allowed modules using a strict check against the file system or a configuration list before importing.

## 2025-05-20 - Insecure Deserialization via Pickle
**Vulnerability:** The `TechnicalAnalyst` class in `core/analysis/technical_analysis.py` uses `pickle.load()` to load ML models from paths defined in configuration. If the config or the file is compromised, this leads to RCE.
**Learning:** Usage of `pickle` for model loading is pervasive in Data Science code but insecure for production systems handling untrusted data.
**Prevention:** Use safer alternatives like `skops` for scikit-learn models, or ensuring strict integrity checks (checksums/signatures) on model files before loading.

## 2025-05-20 - Weak Default Secrets in Code
**Vulnerability:** The Neo4j connection logic defaulted to the password "password" if the environment variable was missing.
**Learning:** Hardcoded fallbacks for secrets, even if intended for "dev convenience," often leak into production environments where env vars might be missed during deployment.
**Prevention:** Remove default values for sensitive credentials. Fail fast (raise Error) or default to `None` to force explicit configuration.

## 2025-12-16 - Hardcoded Flask Secret Key Fallback
**Vulnerability:** The Flask application used the common tutorial pattern `SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'`. This means the application would run with a publicly known secret key if the environment variable was omitted in production, enabling session forgery.
**Learning:** Copy-pasting configuration code from tutorials often introduces insecure defaults. The `or 'value'` idiom is dangerous for secrets.
**Prevention:** Remove default values for secrets in configuration classes. Implement an `init_app` check that explicitly raises a `RuntimeError` if the secret is missing in a non-development environment.

## 2025-12-17 - Unauthenticated Agent Execution
**Vulnerability:** The `/api/agents/<agent_name>/invoke` endpoint allowed any unauthenticated user to execute potentially resource-intensive or sensitive agents.
**Learning:** Endpoints that bridge HTTP to internal command/agent execution patterns are critical high-risk targets that often bypass standard resource access controls.
**Prevention:** Strictly enforce authentication on all "command" or "action" style endpoints.

## 2025-12-18 - Legacy File Shadowing & Error Leakage
**Vulnerability:** `core/api.py` (legacy Flask app) was shadowed by `core.api` (FastAPI package) but remained executable and vulnerable to information leakage via unhandled exceptions.
**Learning:** Naming conflicts between files (`api.py`) and packages (`api/`) can hide legacy code from standard testing/linting tools while leaving it exposed on the filesystem.
**Prevention:** Audit codebase for file/directory name collisions and explicitly deprecate or remove legacy files that are shadowed.

## 2025-12-19 - SQLite SQL Injection & Excessive Exposure
**Vulnerability:** A "SQL Query" tool designed for LLM agents allowed arbitrary `SELECT` queries (including `UNION`-based injection) against the entire database, exposing sensitive tables like `secrets` if they existed.
**Learning:** Checking for `startswith("SELECT")` is insufficient to prevent SQL injection or excessive data exposure. LLM tools that execute code or queries are inherently high-risk.
**Prevention:** Use `sqlite3.set_authorizer` to implement a strict whitelist of allowed tables and actions at the connection level. Combine with URI read-only mode (`?mode=ro`) for defense-in-depth.

## 2025-12-21 - Git Clone Path Traversal & Sync/Async Mismatch
**Vulnerability:** The `GitRepoSubAgent` was vulnerable to path traversal (e.g., `http://example.com/..`) allowing clones outside the target directory. Additionally, the agent defined a synchronous `execute` method while the base class monkey-patched it to be async, causing runtime crashes.
**Learning:** Naive path validation (`split('/')[-1]`) is insufficient for URL-derived paths. Also, complex architecture changes (like monkey-patching base classes for async) can silently break legacy subclasses that don't adhere to the new contract.
**Prevention:** Use `os.path.abspath` and `startswith` (or `commonpath`) to validate paths against a safe root. Ensure subclasses verify compliance with base class contracts, especially when metaclass or `__init__` magic is involved.

## 2025-12-22 - SSRF in Supply Chain Agent
**Vulnerability:** The `SupplyChainRiskAgent` blindly followed URLs provided in its configuration to scrape content, exposing the system to Server-Side Request Forgery (SSRF) against internal services (e.g., cloud metadata, localhost).
**Learning:** Agents that consume "urls" from configuration or user input are prime targets for SSRF. Simply using `requests.get()` without validation is a common oversight.
**Prevention:** Implement a strict `_is_safe_url` validator that checks the URL scheme (http/https) and blocks private/loopback IP addresses using the `ipaddress` module.

## 2024-06-03 - Flask Security Headers
**Vulnerability:** Missing security headers (CSP, HSTS, X-Content-Type-Options) in Flask applications.
**Learning:** Default Flask apps do not include these headers, leaving them vulnerable to XSS and MIME sniffing.
**Prevention:** Always use a middleware or an after_request hook to inject security headers.

### Status and Integration Review
**Status:** Implemented in `core/api.py` and `services/webapp/api.py`. Verified via unit tests.
**Integration:** The headers are applied via a Flask `after_request` hook, ensuring they cover all endpoints served by these applications.
**Relevance:**
- **CSP (`default-src 'self'`):** Critical for preventing XSS in the Adam v23 dashboard, especially given the dynamic rendering of agent outputs.
- **HSTS:** Essential for the production environment where financial data is transmitted.
- **X-Frame-Options:** Prevents the dashboard from being embedded in malicious sites (Clickjacking), protecting the "Mission Control" interface.

## 2024-05-22 - [CRITICAL] XXE in XBRL Handler
**Vulnerability:** Usage of `xml.etree.ElementTree` to parse XBRL files allows XML External Entity (XXE) attacks.
**Learning:** Even internal financial data parsers must treat inputs as untrusted. Standard library XML parsers are often insecure by default.
**Prevention:** Use `defusedxml` for all XML parsing tasks.

## 2024-05-22 - [CRITICAL] Insecure Deserialization (Pickle)
**Vulnerability:** `core/analysis/technical_analysis.py` uses `pickle.load` to load ML models.
**Learning:** Pickle is inherently insecure and allows arbitrary code execution if the file is tampered with.
**Prevention:** Use safer formats like ONNX, Safetensors, or JSON for model serialization. Never unpickle untrusted data.

## 2024-05-22 - [HIGH] Hardcoded Secrets
**Vulnerability:** Potential hardcoded API keys or secrets found in `config/Adam_v22.0_Portable_Config.json` and `tinker_lab/tinker-cookbook/AGENTS.md`.
**Learning:** Secrets in code/config files can be leaked via version control.
**Prevention:** Use environment variables or a secrets manager. Scan commits for high-entropy strings.

## 2024-05-22 - [MEDIUM] Flask Debug Mode
**Vulnerability:** Flags in `final_check_2.txt` suggest Flask might be running with `debug=True` in some contexts.
**Learning:** Debug mode exposes the Werkzeug debugger, which allows arbitrary code execution.
**Prevention:** Ensure `debug=False` is strictly enforced in all production entry points.
## 2025-12-27 - [Hardcoded API Key Placeholder]
**Vulnerability:** Hardcoded API key placeholder ('YOUR_NEWS_API_KEY') in 'core/agents/event_driven_risk_agent.py'.
**Learning:** Placeholder strings for secrets can be dangerous if committed, as they encourage users to edit the file directly, potentially leaking secrets if the file is tracked.
**Prevention:** Always use environment variables for configuration. Agents should fail gracefully or default to mock mode if critical secrets are missing, rather than using placeholder strings.
## 2025-12-28 - Critical RCE in MCP Tools
**Vulnerability:** Found Remote Code Execution (RCE) vulnerabilities in `server/server.py` and `server/mcp_server.py`. The `execute_python_sandbox` tool allowed arbitrary code execution using `exec()`, sometimes even explicitly re-enabling `__builtins__`.
**Learning:** Developers often add "sandbox" features for demos or debugging without realizing the immense security risk. `exec()` in Python is inherently unsafe for untrusted input.
**Prevention:** Implemented a `SecureSandbox` module (`core/security/sandbox.py`) using a "Defense in Depth" strategy: Static Analysis (AST validation), Restricted Globals (whitelisting safe functions), Process Isolation, and Execution Timeouts. This allows useful functionality (like math/logic) while blocking RCE vectors.
## 2024-05-23 - Insecure Deserialization in StateManager
**Vulnerability:** The `StateManager` class in `src/adam/core/state_manager.py` was using `pickle.loads` to deserialize data retrieved from Redis.
**Learning:** Redis is often treated as a trusted data store, but in a microservices environment, it can be a vector for lateral movement. If an attacker compromises a service with Redis access, they can inject malicious payloads to compromise other services.
**Prevention:** Always use safe deserialization methods. We replaced `pickle.loads` with `core.security.safe_unpickler.safe_loads`, which restricts the allowed classes to a safe whitelist (numpy, pandas, torch, etc.).

## 2025-01-04 - Insecure SQL Blacklist
**Vulnerability:** The `LakehouseConnector` in `core/data_access/lakehouse_connector.py` used a blacklist (`if "DROP" in ...`) to prevent SQL injection. This approach is fundamentally flawed as it fails to block other dangerous commands like `INSERT`, `ALTER`, `TRUNCATE`, `GRANT`, or chained commands using semicolons.
**Learning:** Blacklists for security are almost always insufficient because attackers can use synonyms, different case, or commands not on the list.
**Prevention:** Always use a whitelist approach for security validations. In this case, we implemented a custom tokenizer-based SQL validator to strictly enforce `SELECT` queries and prevent multiple statements, handling comments and literals correctly without external dependencies.
