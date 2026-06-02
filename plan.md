1. **Update `src/governance/gatekeeper.py` to check data hashes, source reachability, and add entry/exit gates**:
   - Import `hashlib` and `urllib.request`.
   - In `validate_inference`, compute the SHA-256 hash of the `data` payload using `json.dumps(payload, sort_keys=True).encode("utf-8")`.
   - Verify that the computed hash matches the `content_hash` in the `ProvenanceHeader`.
   - Add a reachability check for `source_data_object` if it looks like a URL. Try fetching it using `urllib.request.urlopen` with a 'Mozilla/5.0' User-Agent header and a short timeout.
   - Implement `entry_gate(self, input_data: Dict[str, Any]) -> Dict[str, Any]` which returns the input data unchanged or does some lightweight preprocessing.
   - Implement `exit_gate(self, inference_output: Dict[str, Any]) -> Dict[str, Any]` which calls `validate_inference`.
2. **Verify `src/governance/gatekeeper.py` modification**:
   - Use `cat src/governance/gatekeeper.py` to confirm the changes were applied correctly.
3. **Update `tests/unit/test_gatekeeper.py`**:
   - Update `tests/unit/test_gatekeeper.py` to mock `urllib.request.urlopen` in all tests (or at least patch it globally) and update `content_hash` to be the actual sha256 of `{"status": "ok", "value": 42}` (which is `b0031fff783bddbdc3707c7c15704944799fdf8d7e69fcdfadb9fef0d4c1b1b1`) in the `get_valid_provenance` function, and compute the actual hash for the fuzz test dynamically.
4. **Verify `tests/unit/test_gatekeeper.py` modification**:
   - Use `cat tests/unit/test_gatekeeper.py` to confirm the changes were applied correctly.
5. **Update `tests/evals/test_fiduciary_fitness.py`**:
   - Update `test_governance_gatekeeper_fuzz`, `test_invalid_provenance_trace`, and `test_valid_inference` in `tests/evals/test_fiduciary_fitness.py` to compute and provide the correct `content_hash` of their respective `payload`/`data` dictionaries and mock `urllib.request.urlopen`.
6. **Verify `tests/evals/test_fiduciary_fitness.py` modification**:
   - Use `cat tests/evals/test_fiduciary_fitness.py` to confirm the changes were applied correctly.
7. **Test evaluation step**:
   - Run tests and verifications using `PYTHONPATH=. pytest tests/unit/test_gatekeeper.py` and `PYTHONPATH=. pytest tests/evals/test_fiduciary_fitness.py`.
8. **Pre commit step**:
   - Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.
9. **Submit**:
   - Call `submit` to commit and push to a branch.
