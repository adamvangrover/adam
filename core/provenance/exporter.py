import json
from core.provenance.lineage import LineageGraph

class ProvJSONLDExporter:
    @staticmethod
    def export(graph: LineageGraph) -> str:
        """
        Exports the lineage graph to W3C PROV-JSONLD format.
        """
        document = {
            "@context": "http://www.w3.org/ns/prov",
            "entity": {k: v.model_dump() for k, v in graph.entities.items()},
            "activity": {k: v.model_dump() for k, v in graph.activities.items()},
            "agent": {k: v.model_dump() for k, v in graph.agents.items()},
            "wasGeneratedBy": graph.wasGeneratedBy,
            "used": graph.used,
            "wasAssociatedWith": graph.wasAssociatedWith,
            "wasDerivedFrom": graph.wasDerivedFrom
        }
        return json.dumps(document, indent=2)
