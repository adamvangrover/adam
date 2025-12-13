# core/engine/neuro_symbolic_planner.py

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
import re
from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END, START
from core.engine.unified_knowledge_graph import UnifiedKnowledgeGraph
from core.engine.states import GraphState, PlanOnGraph

# Fallback for Agent execution (we can inject the real one if needed)
from core.agents.risk_assessment_agent import RiskAssessmentAgent

logger = logging.getLogger(__name__)

class NeuroSymbolicPlanner:
    def __init__(self):
        self.kg = UnifiedKnowledgeGraph()
        # We initialize a default agent for execution of planned steps
        # In a full system, this would be dynamic based on the step 'agent' field.
        try:
            self.default_agent = RiskAssessmentAgent(config={"name": "PlannerExecutor"})
        except Exception:
            self.default_agent = None

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
        graph = self.kg.graph
        
        path_nodes = []
        try:
            path_nodes = nx.shortest_path(graph, source=start_node, target=target_node)
        except (nx.NodeNotFound, nx.NetworkXNoPath) as e:
            logger.warning(f"Path finding failed: {e}. Falling back to default plan.")
            # Fallback Plan if no path found
            return self._generate_fallback_plan(start_node, target_node)

        # 2. Convert path nodes to a structured Symbolic Plan
        match_clauses = [f"(n0 {{name: '{start_node}'}})"]
        steps = []
        raw_path = []

        for i in range(len(path_nodes) - 1):
             u = path_nodes[i]
             v = path_nodes[i+1]

             edge_data = graph.get_edge_data(u, v)
             relation = edge_data.get("relation", "related_to")

             match_clauses.append(f"-[:{relation}]->(n{i+1} {{name: '{v}'}})")
             raw_path.append({"source": u, "relation": relation, "target": v})

             steps.append({
                 "task_id": str(i+1),
                 "agent": "RiskAssessmentAgent", # Dynamic assignment could go here
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

    def _generate_fallback_plan(self, start_node: str, target_node: str) -> Dict[str, Any]:
        """Generates a default plan if KG traversal fails."""
        steps = [
            {
                "task_id": "1",
                "agent": "RiskAssessmentAgent",
                "description": f"Analyze {start_node} explicitly.",
                "cypher_fragment": None
            },
            {
                "task_id": "2",
                "agent": "RiskAssessmentAgent",
                "description": f"Assess impact on {target_node}.",
                "cypher_fragment": None
            }
        ]
        return {"symbolic_plan": "Fallback Plan", "steps": steps, "raw_path": []}

    def parse_natural_language_plan(self, text: str) -> Dict[str, Any]:
        """
        Parses a numbered list of tasks from LLM output (AWO Phase 1).
        Example input:
        1. Ingest data.
        2. Calculate risk.
        """
        steps = []
        # Regex to capture "1. Task description"
        pattern = re.compile(r"^\s*(\d+)\.\s+(.+)$", re.MULTILINE)

        matches = pattern.findall(text)
        if not matches:
             logger.warning("No numbered steps found in text. Returning empty plan.")
             return {"symbolic_plan": "Manual Plan", "steps": []}

        for i, (num, desc) in enumerate(matches):
            steps.append({
                "task_id": str(num),
                "agent": "GeneralAgent", # Or infer from description
                "description": desc.strip(),
                "cypher_fragment": None
            })

        logger.info(f"[NeuroSymbolicPlanner] Parsed {len(steps)} steps from natural language.")
        return {
            "symbolic_plan": "Natural Language Plan",
            "steps": steps,
            "raw_path": []
        }

    def execute_step(self, state: GraphState) -> Dict[str, Any]:
        """
        Executes a single step from the plan.
        """
        plan = state.get("plan")
        if not plan:
            return {"assessment": {"error": "No plan found"}}

        index = state.get("current_task_index", 0)
        steps = plan.get("steps", [])
        
        if index >= len(steps):
            return {"assessment": {"status": "Complete"}}

        step = steps[index]
        description = step.get("description", "Unknown Task")
        logger.info(f"[Planner] Executing Step {index + 1}: {description}")

        # Execute Agent Logic
        # In a real system, we would route to the specific agent in step['agent']
        # Here we simulate/use the default agent
        result_text = f"Executed: {description}"
        if self.default_agent:
             # Just a mock call for now to avoid side effects
             pass

        # Update assessment
        current_assessment = state.get("assessment") or {}
        content = current_assessment.get("content", "")
        content += f"\n- Step {index + 1}: {result_text}"

        return {
            "assessment": {"content": content},
            "current_task_index": index + 1
        }

    def should_continue(self, state: GraphState) -> str:
        """Determines if the graph should continue to the next step or end."""
        plan = state.get("plan")
        index = state.get("current_task_index", 0)
        steps = plan.get("steps", [])

        if index < len(steps):
            return "continue"
        return "end"

    def to_executable_graph(self, plan_data: Dict[str, Any]) -> StateGraph:
        """
        Compiles the symbolic plan into a LangGraph.
        """
        # We create a dynamic graph that loops until the plan is done
        workflow = StateGraph(GraphState)

        # Add the execution node
        workflow.add_node("executor", self.execute_step)
        workflow.set_entry_point("executor")

        # Add conditional edges to loop
        workflow.add_conditional_edges(
            "executor",
            self.should_continue,
            {
                "continue": "executor",
                "end": END
            }
        )

        return workflow.compile()
