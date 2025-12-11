## 2025-12-10 - Flask Error Leakage & Broken Tests
**Vulnerability:** Flask's default error handler leaks exception details (including SQL queries and table names) to the client. Additionally, unit tests were mocking the wrong method (`run_agent` instead of `execute_agent`), causing tests to pass even when the underlying code was broken (returning 500s).
**Learning:** Default Flask configuration is not secure for production. Mocks in tests can mask broken code if they don't match the actual implementation.
**Prevention:** Always implement a custom error handler that returns generic messages. Verify mocks against the actual class interface (e.g., using `autospec=True` or careful review).

## 2025-12-11 - Permissive CORS Configuration
**Vulnerability:** The Flask API was configured with `CORS(app)` without arguments, which defaults to allowing all origins (`*`) and reflecting the origin header. This exposes the API to Cross-Site Request Forgery (CSRF) and data exfiltration from malicious sites.
**Learning:** `flask-cors` is permissive by default ("Allow All") to simplify development, but this is dangerous if not explicitly restricted.
**Prevention:** Always configure `resources` with a specific list of `origins` in `CORS()`, preferably loaded from environment variables to allow flexibility across environments (dev vs prod).
