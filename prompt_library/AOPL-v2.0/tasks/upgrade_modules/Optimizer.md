## Phase 3: THE OPTIMIZER (Performance & Safety)
**Objective**: Audit for bottlenecks, resource leaks, and security vulnerabilities. Optimize for time and space complexity. Fortify error handling to ensure the application fails gracefully.

### Instructions:
1.  **Algorithmic Optimization**: Review loops, sorting, and search operations. Can the time complexity (Big O) be improved (e.g., changing O(N^2) to O(N log N) or O(N))? Can space complexity be reduced?
2.  **Resource Management**: Check for potential resource leaks (e.g., unclosed files, network connections, database cursors). Ensure proper use of context managers (e.g., `with` statements in Python).
3.  **Security Audit**: Scan for common vulnerabilities. Are inputs properly validated and sanitized? Are database queries safe from injection (e.g., using parameterized queries)? Are sensitive data handled securely?
4.  **Error Handling**: Replace broad `try-except` blocks with specific exception catching. Ensure that errors are logged meaningfully and that the application degrades or fails gracefully without exposing internal state or crashing abruptly.
5.  **Caching/Memoization**: Introduce caching mechanisms for expensive or frequently called functions if appropriate.
6.  **Verify Tests**: Add new tests to specifically target the fortified error handling and edge cases. Run the full suite.