from typing import Dict, Any, List
from datetime import datetime, timezone

class ProvJSONLDExporter:
    """
    Exports decision lineage as a directed acyclic graph mapping to W3C PROV-O elements,
    specifically incorporating PROV-AGENT extensions to treat AI actions as first-class components.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    def export_lineage_graph(
        self,
        activity_id: str,
        used_entity_hashes: List[str],
        generated_entity_hash: str,
        context_hash: str
    ) -> Dict[str, Any]:
        """
        Generates a W3C PROV-JSONLD structure mapping the execution.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        graph = {
            "@context": {
                "prov": "http://www.w3.org/ns/prov#",
                "adam": "http://adam-os.local/schema#"
            },
            "@graph": [
                {
                    "@id": f"adam:agent:{self.agent_id}",
                    "@type": "prov:Agent"
                },
                {
                    "@id": f"adam:activity:{activity_id}",
                    "@type": "prov:Activity",
                    "prov:startedAtTime": timestamp,
                    "prov:wasAssociatedWith": {"@id": f"adam:agent:{self.agent_id}"},
                    "prov:used": [{"@id": f"adam:entity:{h}"} for h in used_entity_hashes] + [{"@id": f"adam:entity:{context_hash}"}]
                },
                {
                    "@id": f"adam:entity:{generated_entity_hash}",
                    "@type": "prov:Entity",
                    "prov:wasGeneratedBy": {"@id": f"adam:activity:{activity_id}"},
                    "prov:wasDerivedFrom": [{"@id": f"adam:entity:{h}"} for h in used_entity_hashes]
                }
            ]
        }

        return graph
