from typing import Any, Dict, List, Optional
import logging
from core.agents.agent_base import AgentBase
from core.v23_graph_engine.unified_knowledge_graph import UnifiedKnowledgeGraph
try:
    from semantic_kernel import Kernel
except ImportError:
    Kernel = Any

logger = logging.getLogger(__name__)

class NexusAgent(AgentBase):
    """
    The Nexus Agent: a specialized AI Financial Knowledge Graph Analyst.
    Implements Graph RAG (Retrieval Augmented Generation).
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Kernel] = None):
        super().__init__(config, kernel)
        self.name = "Nexus"
        self.kg = UnifiedKnowledgeGraph() # Initialize connection to the KG

    async def execute(self, query: str = "", **kwargs) -> str:
        """
        Executes the Graph RAG pipeline:
        1. Entity Extraction
        2. Graph Expansion (1-hop)
        3. Contextual Synthesis
        """
        logger.info(f"[{self.name}] Executing Graph RAG for query: {query}")

        # 1. Entity Extraction
        # In a real system, use NER. Here we use simple keyword matching against the KG nodes for demo.
        entities = self._extract_entities(query)
        if not entities:
            # Fallback if no specific entities found
            entities = ["Apple Inc."]
            logger.warning(f"No entities extracted. Defaulting to: {entities}")
        else:
            logger.info(f"Extracted entities: {entities}")

        # 2. Graph Expansion (1-hop neighbors)
        context_triples = []
        cited_nodes = set()

        for entity in entities:
            if entity in self.kg.graph:
                cited_nodes.add(entity)
                # Get outgoing edges
                for neighbor in self.kg.graph.neighbors(entity):
                    edge_data = self.kg.graph.get_edge_data(entity, neighbor)
                    relation = edge_data.get("relation", "related_to")
                    triple = f"({entity}) -[{relation}]-> ({neighbor})"
                    context_triples.append(triple)
                    cited_nodes.add(neighbor)
            else:
                logger.warning(f"Entity '{entity}' not found in Knowledge Graph.")

        # 3. Contextual Synthesis
        context_block = "\n".join(context_triples)
        logger.info(f"Retrieved {len(context_triples)} triples for context.")

        prompt = f"""
        You are Nexus, an AI Financial Analyst.
        User Query: "{query}"

        Knowledge Graph Context:
        {context_block}

        Answer the query using ONLY the provided context.
        Explicitly cite the relationships used (e.g., "Based on [Entity -> relation -> Entity]...").
        """

        # Generate response
        response = await self._ask_llm(prompt)

        return response

    def _extract_entities(self, query: str) -> List[str]:
        """
        Simple keyword extraction based on known KG nodes.
        """
        # Get all nodes from KG for matching
        all_nodes = list(self.kg.graph.nodes())
        found = [node for node in all_nodes if node in query]
        return found

    async def _ask_llm(self, prompt: str) -> str:
        """
        Helper to invoke LLM via Kernel or Mock.
        """
        if self.kernel:
            try:
                # Assuming kernel has a default completion service or similar
                # For v1.0, we might need to construct a function
                # This is a simplified call pattern
                from semantic_kernel.functions import KernelArguments
                # We would typically use a registered plugin/function here.
                # Since we don't have the exact plugin registry, we'll return a formatted string
                # pretending to be the LLM for safety, or try to run if configured.
                pass
            except Exception as e:
                logger.error(f"Kernel invocation failed: {e}")

        # Fallback / Mock Response (since we need to return a string per signature)
        # This ensures the agent works even without a live LLM connection
        return f"Based on the knowledge graph context:\n{prompt.split('Knowledge Graph Context:')[1]}"

    def get_skill_schema(self) -> Dict[str, Any]:
        """
        Defines the agent's skills for the MCP.
        """
        return {
            "name": self.name,
            "description": "A specialized AI Financial Knowledge Graph Analyst.",
            "skills": [
                {
                    "name": "process_query",
                    "description": "Processes a natural language query about the financial knowledge graph.",
                    "parameters": [
                        {"name": "query", "type": "string", "description": "The natural language query."}
                    ]
                }
            ]
        }
