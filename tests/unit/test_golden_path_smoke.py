"""The golden-path smoke script runs offline and emits a valid PROV-O document."""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("structlog")
pytest.importorskip("json_logic")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "golden_path_smoke.py"


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location("golden_path_smoke", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_golden_path_emits_valid_prov(smoke, tmp_path, monkeypatch):
    out = tmp_path / "prov.jsonld"
    monkeypatch.setattr(sys, "argv", ["golden_path_smoke.py", "--out", str(out)])
    assert asyncio.run(smoke.main()) == 0
    doc = json.loads(out.read_text())
    types = [n["@type"] for n in doc["@graph"]]
    assert types.count("prov:Activity") == 4 and types.count("prov:Entity") == 4
    bundle = next(n for n in doc["@graph"] if n["@type"] == "prov:Bundle")
    assert bundle["adam:trace"], "System 2 provenance_trace must be non-empty"


def test_validate_prov_rejects_dangling_reference(smoke):
    doc = smoke.prov_document([{"id": "adam:x", "label": "x"}], [])
    doc["@graph"][-1]["prov:wasDerivedFrom"] = {"@id": "adam:missing"}
    with pytest.raises(AssertionError):
        smoke.validate_prov(doc)
