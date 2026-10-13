import json
import os
import pytest


def test_memory_node_valid_fixture():
    fixture_path = "evals/agent_utilities/fixtures/memory_node_valid.json"
    assert os.path.exists(fixture_path)
    with open(fixture_path, "r", encoding="utf-8") as f:
        node = json.load(f)
    assert node["schema_version"] == "1.0"
    assert node["tier"] in ["M0", "M1", "M2", "M3"]
    assert len(node["content_hash"]) == 64


def test_memory_node_invalid_m3_unverified():
    fixture_path = "evals/agent_utilities/fixtures/memory_node_invalid.json"
    assert os.path.exists(fixture_path)
    with open(fixture_path, "r", encoding="utf-8") as f:
        node = json.load(f)
    # Invalid fixture attempts M3 admission with UNVERIFIED status
    assert node["tier"] == "M3"
    assert node["epistemic_status"] == "UNVERIFIED"
