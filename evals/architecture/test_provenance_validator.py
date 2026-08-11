from core.provenance.prov import ProvEntity, ProvActivity
from core.provenance.lineage import LineageGraph

def test_lineage_graph_structure():
    graph = LineageGraph()
    e1 = ProvEntity(id="e1")
    a1 = ProvActivity(id="a1")

    graph.add_entity(e1)
    graph.add_activity(a1)
    graph.record_generation(e1.id, a1.id)

    assert "e1" in graph.entities
    assert len(graph.wasGeneratedBy) == 1
    assert graph.wasGeneratedBy[0]["entity"] == "e1"
