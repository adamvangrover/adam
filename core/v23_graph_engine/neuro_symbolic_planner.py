# core/v23_graph_engine/neuro_symbolic_planner.py

"""
Implements the Plan-on-Graph (PoG) framework for verifiable, grounded planning.

This module replaces the generative 'WorkflowCompositionSkill' from v22.0.
Instead of generating a plan from parametric knowledge, this planner discovers
a valid reasoning path by traversing the Unified Knowledge Graph (FIBO + PROV-O).

This ensures that every workflow is grounded in a verifiable, symbolic scaffold
before any generative agents are invoked.
"""

import logging
import networkx as nx
from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END, START
from core.v23_graph_engine.unified_knowledge_graph import UnifiedKnowledgeGraph

# Attempt to import GraphState from the POC, or define a compatible one
try:
    from core.system.v23_graph_engine.adaptive_system_poc import GraphState, PlanOnGraph
except ImportError:
    # Fallback definition
    class PlanOnGraph(TypedDict):
        id: str
        steps: List[Dict[str, Any]]
        is_complete: bool

    class GraphState(TypedDict):
        request: str
        plan: Optional[PlanOnGraph]
        current_task_index: int
        assessment: Optional[Dict[str, Any]]
        critique: Optional[Dict[str, Any]]
        human_feedback: Optional[str]
        iteration: int
        max_iterations: int

logger = logging.getLogger(__name__)

class NeuroSymbolicPlanner:
    def __init__(self):
        self.kg = UnifiedKnowledgeGraph()

    def discover_plan(self, start_node: str, target_node: str) -> Dict[str, Any]:
        """
        Discovers a valid reasoning path in the KG and returns a symbolic plan
        containing the actual Cypher query.

        Args:
            start_node: The starting entity (e.g., "Apple Inc.")
            target_node: The target concept (e.g., "CreditRating")

        Returns:
            Dict containing the 'symbolic_plan' (Cypher query) and 'steps'.
        """
        logger.info(f"[NeuroSymbolicPlanner] Discovering path from '{start_node}' to '{target_node}'...")
        
        # 1. Pathfinding: Implement algorithm (Dijkstra) directly on the KG edges
        # We access the underlying graph structure to find valid transitions.
        graph = self.kg.graph
        
        path_nodes = []
        try:
            # Dijkstra Algorithm (unweighted shortest path)
            path_nodes = nx.shortest_path(graph, source=start_node, target=target_node)
        except nx.NodeNotFound as e:
            logger.warning(f"Entity not found in KG: {e}")
            return {"symbolic_plan": None, "steps": []}
        except nx.NetworkXNoPath:
            logger.warning(f"No path found between {start_node} and {target_node}.")
            return {"symbolic_plan": None, "steps": []}

        # 2. Convert path nodes to a structured Symbolic Plan (Cypher Query)
        # Path is a list of nodes: [u, v, w, ...]

        match_clauses = []
        match_clauses.append(f"(n0 {{name: '{start_node}'}})")

        steps = []
        raw_path = []

        for i in range(len(path_nodes) - 1):
             u = path_nodes[i]
             v = path_nodes[i+1]

             # Query edge attributes from the KG to get the specific relationship
             edge_data = graph.get_edge_data(u, v)
             relation = edge_data.get("relation", "related_to")

             # Add to Cypher pattern: -[:rel]->(n{i+1})
             match_clauses.append(f"-[:{relation}]->(n{i+1} {{name: '{v}'}})")

             # Record raw path info
             raw_path.append({"source": u, "relation": relation, "target": v})

             # Create an actionable step for the plan
             steps.append({
                 "task_id": str(i+1),
                 "agent": "RiskAssessmentAgent",
                 "description": f"Verify relationship: {u} -[{relation}]-> {v}",
                 "cypher_fragment": f"MATCH (a {{name: '{u}'}})-[:{relation}]->(b {{name: '{v}'}}) RETURN a, b"
             })

        cypher_query = "MATCH path = " + "".join(match_clauses) + " RETURN path"
        logger.info(f"Generated Cypher Plan: {cypher_query}")

        return {
            "symbolic_plan": cypher_query,
            "steps": steps,
            "raw_path": raw_path
        }

    def execute(self, state: GraphState) -> Dict[str, Any]:
        """
        Execution node for the AdaptiveSystemGraph.
        Parses the user request, discovers a plan, and updates the state.
        """
        request = state.get("request", "")
        logger.info(f"[NeuroSymbolicPlanner] Processing request: {request}")

        # Simple Entity Extraction
        start_node = "Apple Inc."
        target_node = "CreditRating"
        
        if "Tesla" in request:
            start_node = "Tesla Inc."
        
        # Discover the plan
        discovery_result = self.discover_plan(start_node, target_node)

        symbolic_plan_query = discovery_result.get("symbolic_plan")
        steps = discovery_result.get("steps", [])
        
        plan: PlanOnGraph = {
            "id": "plan-generated-001",
            "steps": steps,
            "is_complete": False
        }

        # Inject query
        plan["cypher_query"] = symbolic_plan_query

        if not symbolic_plan_query:
            logger.error("Failed to generate a symbolic plan.")

        return {"plan": plan, "current_task_index": 0}

    def to_executable_graph(self, symbolic_plan: List[Dict[str, str]]):
        """
        Compiles the symbolic plan into a LangGraph.
        """
        pass
