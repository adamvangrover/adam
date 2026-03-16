import logging
import asyncio
from typing import Dict, Any

from core.engine.memory.vector_memory import VectorMemoryManager
from core.engine.memory.graph_memory import GraphMemoryManager

class MemoryOrchestrator:
    """
    Synthesizes and manages contextual queries across both the active
    Vector Database (Semantic RAG) and Graph Database (Entity Relationships).
    
    This acts as the bridge injecting pre-computed context into the System 2
    LangGraphs before they generate their financial models.
    """
    
    def __init__(self):
        logging.info("Initializing MemoryOrchestrator...")
        self.vector_db = VectorMemoryManager()
        self.graph_db = GraphMemoryManager()

    async def get_total_context(self, entity: str, thematic_query: str) -> Dict[str, Any]:
        """
        Runs concurrent operations to build a complete situational awareness map
        for a given entity and thematic query.
        """
        logging.info(f"MemoryOrchestrator building Total Context for [{entity}]: '{thematic_query}'")
        
        # We can run I/O bound DB queries concurrently
        vector_task = asyncio.to_thread(self.vector_db.semantic_search, thematic_query, k=3)
        graph_task = asyncio.to_thread(self.graph_db.query_relationships, entity, radius=2)
        
        vector_results, graph_results = await asyncio.gather(vector_task, graph_task)
        
        # Assemble Synthesis
        context_envelope = {
            "target_entity": entity,
            "query": thematic_query,
            "semantic_history": [],
            "structural_relationships": graph_results
        }
        
        # Flatten vector text for the LLM State context
        for v in vector_results:
            context_envelope["semantic_history"].append(v["text"])
            
        logging.info(f"Context Map Built. Recovered {len(vector_results)} semantic blocks and {len(graph_results)} relational edges.")
        return context_envelope
        
    def prepopulate_memory(self, path_to_file: str, doc_id: str):
        """
        Utility function to ingest legacy markdown files into the Active Vector Memory.
        """
        try:
            with open(path_to_file, 'r') as f:
                content = f.read()
                
            self.vector_db.ingest_document(
                document_id=doc_id, 
                text=content, 
                metadata={"source": path_to_file, "type": "legacy_report"}
            )
            logging.info(f"Prepopulated {path_to_file} into Active Memory.")
        except Exception as e:
            logging.error(f"Failed to prepopulate memory from {path_to_file}: {e}")
