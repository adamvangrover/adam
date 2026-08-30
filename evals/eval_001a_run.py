import json
import os

def test_ledger_schema():
    ledger_file = "data/institutional_ai_training_ledger.jsonl"
    schema_file = "schemas/institutional_ai_training_ledger.json"

    assert os.path.exists(ledger_file), f"File {ledger_file} not found"
    assert os.path.exists(schema_file), f"File {schema_file} not found"

    with open(schema_file, 'r') as f:
        schema = json.load(f)

    with open(ledger_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = json.loads(line)
        for key in schema['items']['required']:
            assert key in data, f"Required key {key} not found in {data}"

def test_metrics_in_markdown():
    file_path = "newsletters/market_mayhem_20260829.md"
    assert os.path.exists(file_path), f"File {file_path} not found"

    with open(file_path, 'r') as f:
        content = f.read()

    assert "7,711.76" in content, "S&P 500 at 7,711.76 not found"
    assert "4.73%" in content, "10-Year Yield at 4.73% not found"
    assert "88.22" in content, "Brent Crude at 88.22 not found"
    assert "78,255" in content, "Bitcoin at 78,255 not found"

if __name__ == "__main__":
    test_ledger_schema()
    test_metrics_in_markdown()
    print("All tests passed.")