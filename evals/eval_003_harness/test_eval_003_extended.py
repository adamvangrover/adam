import pytest
import json
from evals.eval_003_harness.eval_003_core import Eval003Harness
from src.pdil.provenance.context_freezer import ContextFreezer
from src.pdil.provenance.prov_jsonld_exporter import ProvJSONLDExporter
from src.pdil.transport.event_spine import EventEnvelope, EventSpineSimulation

def test_context_freezing():
    harness = Eval003Harness()
    freezer = ContextFreezer()

    context = {"market_state": "volatile", "prices": [1, 2, 3]}
    frozen = freezer.freeze_context(context)

    assert freezer.verify_frozen_context(frozen) == True

    # Tamper with the frozen context
    frozen["frozen_data"]["market_state"] = "calm"
    assert freezer.verify_frozen_context(frozen) == False

    harness.add_result("provenance_extended", True, "Context freezing cryptographic integrity verified")
    assert harness.certify() == True

def test_prov_jsonld_structure():
    harness = Eval003Harness()
    exporter = ProvJSONLDExporter("agent_01")

    graph = exporter.export_lineage_graph("act_01", ["entity_a", "entity_b"], "entity_c", "hash_context")

    # Basic semantic structure validation
    assert "@context" in graph
    assert "@graph" in graph

    entities = [item for item in graph["@graph"] if item.get("@type") == "prov:Entity"]
    assert len(entities) == 1
    assert entities[0]["@id"] == "adam:entity:entity_c"

    harness.add_result("provenance_extended", True, "PROV-JSONLD semantic structure verified")
    assert harness.certify() == True

def test_event_spine_idempotency():
    harness = Eval003Harness()
    spine = EventSpineSimulation()

    payload = {"trade": "BUY", "amount": 100}
    context_hash = "abc123hash"

    env1 = EventEnvelope(payload, context_hash, "trades.execute")
    env2 = EventEnvelope(payload, context_hash, "trades.execute")

    # First publish should succeed
    assert spine.publish(env1) == True

    # Second publish with same deterministic payload/context should be silently discarded
    assert spine.publish(env2) == False

    harness.add_result("transport", True, "Event idempotency and deduplication window verified")
    assert harness.certify() == True
