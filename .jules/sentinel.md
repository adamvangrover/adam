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
