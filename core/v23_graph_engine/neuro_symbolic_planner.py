import logging
from typing import List, Dict, Any
from core.schemas.v23_5_schema import ExecutionPlan, PlanStep, GraphNode

logger = logging.getLogger(__name__)

class NeuroSymbolicPlanner:
    """
    Implements RAG-Guided Subgraph Retrieval and Planning.
    Moving from static 'shortest_path' to LLM-driven graph traversal.
    """

    def __init__(self, kg_client=None, llm_client=None):
        self.kg_client = kg_client # e.g., Neo4j connection
        self.llm_client = llm_client
        logger.info("NeuroSymbolicPlanner initialized.")

    def generate_plan(self, query: str, context: Dict[str, Any] = None) -> ExecutionPlan:
        """
        Generates a multi-step execution plan based on the user query.
        """
        # 1. Entity Extraction (NER)
        entities = self._extract_entities(query)

        # 2. Vector Anchoring (Find relevant nodes in KG)
        anchors = self._find_anchors(entities)

        # 3. Dynamic Pathfinding (LLM + Cypher)
        # In a real impl, we would generate Cypher here.
        # For now, we simulate the 'Intelligent' planning structure.

        steps = []
        if not anchors:
            # Fallback for when no specific entities are found in KG
            steps.append(PlanStep(
                step_id="step_1",
                action="GENERAL_SEARCH",
                target_entity="Market",
                parameters={"query": query},
                rationale="No specific entities anchored in KG."
            ))
        else:
            for entity in anchors:
                steps.append(PlanStep(
                    step_id=f"step_fetch_{entity.id}",
                    action="FETCH_FUNDAMENTALS",
                    target_entity=entity.label,
                    parameters={"ticker": entity.id},
                    rationale=f"Retrieve core data for {entity.label}"
                ))
                steps.append(PlanStep(
                    step_id=f"step_risk_{entity.id}",
                    action="ASSESS_RISK",
                    target_entity=entity.label,
                    parameters={"context": "Supply Chain"},
                    rationale=f"Analyze supply chain risks for {entity.label}"
                ))

        return ExecutionPlan(
            plan_id="gen_plan_001",
            steps=steps,
            estimated_complexity="MEDIUM"
        )

    def _extract_entities(self, query: str) -> List[str]:
        """
        Uses NER to extract entities from query.
        """
        # Placeholder for Spacy/LLM NER
        # Naive implementation for demo
        known_entities = ["AAPL", "Apple", "TSLA", "Tesla", "NVDA", "Nvidia", "BYD"]
        found = []
        for word in query.split():
            clean_word = word.strip(".,?!")
            if clean_word in known_entities:
                found.append(clean_word)
        return found

    def _find_anchors(self, entities: List[str]) -> List[GraphNode]:
        """
        Maps extracted strings to GraphNodes using Vector Search or Fuzzy Match.
        """
        # Mock KG lookup
        anchors = []
        for ent in entities:
            # Simulate resolving "Tesla" to "TSLA"
            if ent in ["Tesla", "TSLA"]:
                anchors.append(GraphNode(id="TSLA", label="Tesla Inc.", properties={"sector": "Auto"}))
            elif ent in ["Apple", "AAPL"]:
                anchors.append(GraphNode(id="AAPL", label="Apple Inc.", properties={"sector": "Tech"}))
            elif ent in ["BYD"]:
                anchors.append(GraphNode(id="1211.HK", label="BYD Company", properties={"sector": "Auto"}))
        return anchors
