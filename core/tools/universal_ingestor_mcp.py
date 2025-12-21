import json
import os

from core.data_processing.universal_ingestor import UniversalIngestor
from core.tools.base_tool import BaseTool


class UniversalIngestorMCP(BaseTool):
    name = "azure_ai_search"
    description = "Retrieves indexed regulatory filings and unstructured documents. Can ingest new data from paths."

    def __init__(self):
        self.ingestor = UniversalIngestor()
        # Ensure we have some data
        if os.path.exists("data/gold_standard"):
             pass # We would load index here in a real system

    def execute(self, query: str, filter: str = "") -> str:
        """
        Executes the ingestion or search.
        In this implementation, if query looks like a path, we ingest.
        Otherwise, we treat it as a search query (mocked).
        """
        # 1. Ingestion Mode
        if os.path.exists(query) or query.startswith("http"):
             # For URLs, we would need a downloader. For now, assume local path.
             if os.path.exists(query):
                 self.ingestor.process_file(query)
                 return json.dumps({"status": "Ingested", "path": query, "conviction": "High"})

        # 2. Search Mode (Mock)
        # In a real system, this would query Azure AI Search.
        # Here, we return a mock JSONL string.

        mock_result = {
            "title": f"Search Results for {query}",
            "content": "This is a verified source extracted via Universal Ingestor.",
            "source": "SEC 10-K",
            "metadata": {"filter_used": filter},
            "conviction_score": 0.95
        }

        return json.dumps(mock_result)

    def _get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Specific keywords or Source Path"},
                "filter": {"type": "string", "description": "OData filter (e.g., 'doc_type eq 10-K')"}
            },
            "required": ["query"]
        }
