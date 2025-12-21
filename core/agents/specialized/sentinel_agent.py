import logging
from typing import Dict, Any, Optional
from core.agents.agent_base import AgentBase
from core.v23_graph_engine.odyssey_knowledge_graph import OdysseyKnowledgeGraph

logger = logging.getLogger(__name__)


class SentinelAgent(AgentBase):
    """
    The Data Integrity Guardian.
    Responsibility: Ingestion, Extraction, Validation against FIBO Schema.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Any] = None, graph: Optional[OdysseyKnowledgeGraph] = None):
        super().__init__(config, kernel=kernel)
        self.graph = graph or OdysseyKnowledgeGraph()

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Executes the Sentinel workflow.
        Expected kwargs: document_path (str) or entity_data (dict)
        """
        doc_path = kwargs.get("document_path")
        entity_data = kwargs.get("entity_data")

        if entity_data:
            return self.process_entity(entity_data)
        elif doc_path:
            return self.process_document(doc_path)
        else:
            return {"error": "No input provided (document_path or entity_data)"}

    def process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Ingests a document, extracts data, validates, and loads into graph.
        """
        logger.info(f"Sentinel processing {document_path}")

        # 1. Extraction (Mocked for now)
        extracted_data = self._mock_extract(document_path)

        # 2. Validation & Ingestion
        return self.process_entity(extracted_data)

    def process_entity(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Check for critical data failure (e.g., missing revenue if it was a mandatory field)
            # Schema validation handles checking required fields defined in schema.

            self.graph.ingest_odyssey_entity(entity_data)
            logger.info("Entity successfully validated and ingested.")
            return {"status": "success", "entity_id": entity_data.get("@id"), "data": entity_data}
        except ValueError as e:
            logger.error(f"CRITICAL DATA FAILURE: {e}")
            return {"status": "error", "error": str(e), "flag": "FLAG_DATA_MISSING"}

    def _mock_extract(self, doc_path: str) -> dict:
        # Mock extraction logic
        return {
            "@id": f"urn:fibo:be-le-cb:Corporation:US-Mock-{doc_path}",
            "@type": "fibo-be-le-cb:Corporation",
            "legalName": "Mock Corp Inc.",
            "hasCreditFacility": [
                {
                    "@id": "urn:fibo:loan:TermLoanB-001",
                    "facilityType": "Term Loan B",
                    "hasJCrewBlocker": True,
                    "interestCoverageRatio": 2.5
                }
            ]
        }
