import json
import pytest
import os

def test_adam_manifest_exists():
    assert os.path.exists("adam.manifest.json")

def test_adam_manifest_content():
    with open("adam.manifest.json") as f:
        manifest = json.load(f)

    assert manifest["version"] == "Adam OS vNext"
    assert manifest["architecture"] == "Multi-Plane Topology"
    assert "core_paradigms" in manifest
    assert "governance_substrate" in manifest
    assert "context_freezing" in manifest
    assert "polyglot_runtime_topology" in manifest
    assert "model_context_protocol" in manifest
    assert "w3c_provo_integration" in manifest

def test_invariants_exist():
    with open("adam.manifest.json") as f:
        manifest = json.load(f)

    assert "invariants" in manifest
    invariants = manifest["invariants"]
    # Verify we have at least invariant A through J
    for key in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        assert key in invariants

def test_execution_domains_exist():
    with open("adam.manifest.json") as f:
        manifest = json.load(f)

    assert "execution_domains" in manifest
