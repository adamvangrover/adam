# Learnings from Session

## Dependency Management
- **Issue:** Requirements file contained hallucinated versions of core libraries (numpy, pandas, scipy).
- **Solution:** Manually verified and pinned to latest stable versions. 'numpy<2.0.0' is critical for compatibility.

## Testing Strategy
- **Issue:** 'tests/api/test_service_state.py' failed with an obscure 'wrapping' keyword argument error in 'unittest.mock'.
- **Solution:** As per user instruction ('skip and verify'), the test was skipped using '@pytest.mark.skip' and logged in 'docs/outstanding_errors.md' to unblock deployment.

## Frontend
- **Issue:** Build failed due to missing 'recharts' dependency.
- **Solution:** Added 'recharts' to 'package.json'.
