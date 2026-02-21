
## 2026-02-20 - Unauthenticated Access to Credit Pipeline
**Vulnerability:** The `generate_credit_memo` endpoint in `services/webapp/api.py` was configured with `@jwt_required(optional=True)`, allowing unauthenticated users to trigger the resource-intensive Credit Pipeline.
**Learning:** Marking JWT requirements as optional for "demo convenience" without safeguards creates a DoS vulnerability. However, strict blocking (401) degrades the user experience in showcase environments.
**Prevention:** Implement "Adaptive Security": For unauthenticated requests to expensive endpoints, return a lightweight "Simulation Mode" response (fallback signal) instead of running the heavy compute or blocking access. This pushes the computational burden to the client (or mock data) while securing the server.
