import json
import os

def test_manifest_exists():
    assert os.path.exists("adam.manifest.json"), "Machine-readable manifest must exist at root."

def test_manifest_structure():
    with open("adam.manifest.json") as f:
        data = json.load(f)
    assert "architecture" in data
    assert "invariants" in data

    invariants = data["invariants"]
    assert "policy_before_mutation" in invariants
    assert "provenance_before_commit" in invariants
    assert "frozen_context" in invariants
    assert "deterministic_numeric_execution" in invariants
