import json
import os
import tempfile
from scripts.ml_data_flywheel import MLDataFlywheel

def test_sanitize_data():
    flywheel = MLDataFlywheel(log_path="dummy.log", db_path="dummy.jsonl")

    # Test email sanitization
    text_with_email = "User test@example.com experienced LLM_EVAL_FAILED"
    sanitized = flywheel.sanitize_data(text_with_email)
    assert "test@example.com" not in sanitized
    assert "[REDACTED_EMAIL]" in sanitized

    # Test SSN sanitization
    text_with_ssn = "Customer SSN is 123-45-6789 resulting in EDGE_CASE_EXCEPTION"
    sanitized = flywheel.sanitize_data(text_with_ssn)
    assert "123-45-6789" not in sanitized
    assert "[REDACTED_SSN]" in sanitized

    # Test API key sanitization
    text_with_key = "LLM_EVAL_FAILED with key sk-abcdef1234567890abcdef1234567890"
    sanitized = flywheel.sanitize_data(text_with_key)
    assert "sk-abcdef1234567890abcdef1234567890" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized

def test_capture_failures_and_edge_cases():
    # Create temporary files for log and db
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as log_file:
        log_file.write("Normal info log entry\n")
        log_file.write("LLM_EVAL_FAILED: prompt failed to parse due to sk-secretkey12345678901234567890123\n")
        log_file.write("EDGE_CASE_EXCEPTION: division by zero for user@example.com\n")
        log_path = log_file.name

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as db_file:
        db_path = db_file.name

    try:
        flywheel = MLDataFlywheel(log_path=log_path, db_path=db_path)
        flywheel.capture_failures_and_edge_cases()

        # Verify db contents
        with open(db_path, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 2

        # Verify first entry (LLM_EVAL_FAILED)
        entry1 = json.loads(lines[0])
        assert entry1["type"] == "LLM_EVAL_FAILED"
        assert "[REDACTED_API_KEY]" in entry1["raw_log_sanitized"]
        assert "sk-" not in entry1["raw_log_sanitized"]

        # Verify second entry (EDGE_CASE_EXCEPTION)
        entry2 = json.loads(lines[1])
        assert entry2["type"] == "EDGE_CASE_EXCEPTION"
        assert "[REDACTED_EMAIL]" in entry2["raw_log_sanitized"]
        assert "user@example.com" not in entry2["raw_log_sanitized"]

    finally:
        os.remove(log_path)
        os.remove(db_path)
