# 001: Security Hardening & Resilience - Operation Green Light

## Status
Accepted

## Context
The Adam v23.5 codebase required significant hardening to meet enterprise security standards and ensure operational resilience in diverse environments (including those without `langgraph`).

## Decision
1.  **Cryptography**: Replaced all instances of MD5 hashing with SHA-256 to mitigate collision vulnerabilities.
2.  **Web Security**:
    -   Disabled Flask `debug` mode in production-ready files (`ui_backend.py`).
    -   Enabled Jinja2 `autoescape` to prevent XSS attacks in generated newsletters.
3.  **Network Resilience**: Enforced timeouts (30s) on all external API requests (`requests.get`) to prevent hanging threads.
4.  **SQL Safety**: Implemented validation for dynamic SQL queries in `MCPRegistry` to prevent injection.
5.  **Graceful Degradation**: Wrapped `langgraph` imports in `try/except` blocks across all graph engines. If the library is missing, the system now logs a warning and disables the specific graph feature rather than crashing at startup.
6.  **Type Safety**: Relaxed `TypedDict` strictness (`total=False`) in `core/engine/states.py` to allow for progressive state construction in asynchronous graph workflows.

## Consequences
-   **Pros**: Significantly reduced attack surface; system is now runnable in minimal environments (CI/CD without optional deps); improved type safety for partial updates.
-   **Cons**: Partial state updates might mask logic errors if keys are permanently missing; strict validation moved to runtime.
