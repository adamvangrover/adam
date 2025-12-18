# core/engine/neuro_symbolic_planner.py

import logging
import networkx as nx
import re
from typing import List, Dict, Any, Optional, Set
from langgraph.graph import StateGraph, END, START

# Internal imports
from core.engine.unified_knowledge_graph import UnifiedKnowledgeGraph
from core.engine.states import GraphState

logger = logging.getLogger(__name__)

class NeuroSymbolicPlanner:
    """
    Implements the Plan-on-Graph (PoG) framework for verifiable, grounded planning.
    
    Bridge Pattern:
    - Symbolic Side: Uses NetworkX/Neo4j to find valid reasoning paths.
    - Neural Side: Uses LLMs to parse unstructured requests into directed graphs.
    """

    def __init__(self):
        self.kg = UnifiedKnowledgeGraph()
        self.default_agent = None
        
        # Lazy load the executor agent to avoid circular imports during init
        try:
            from core.agents.risk_assessment_agent import RiskAssessmentAgent
            self.default_agent = RiskAssessmentAgent(config={"name": "PlannerExecutor"})
            logger.info("RiskAssessmentAgent loaded successfully.")
        except ImportError:
            logger.warning("RiskAssessmentAgent not found. Planner running in 'Simulated Execution' mode.")

    # -------------------------------------------------------------------------
    # 1. SYMBOLIC DISCOVERY (Graph Traversal)
    # -------------------------------------------------------------------------

    def discover_plan(self, start_node: str, target_node: str) -> Dict[str, Any]:
        """
        Discovers a grounded reasoning path in the KG.
        
        Algorithm: Unweighted Shortest Path (BFS/Dijkstra).
        Why: Ensures the shortest logical distance between a company and a risk factor.
        """
        logger.info(f"[PoG] Discovering path: '{start_node}' -> '{target_node}'")
        
        graph = self.kg.graph
        
        # 1. Pathfinding
        try:
            path_nodes = nx.shortest_path(graph, source=start_node, target=target_node)
        except (nx.NodeNotFound, nx.NetworkXNoPath) as e:
            logger.warning(f"[PoG] Path finding failed: {e}. Reverting to Fallback.")
            return self._generate_fallback_plan(start_node, target_node)

        # 2. Path to Plan Conversion
        match_clauses = [f"(n0 {{name: '{start_node}'}})"]
        steps = []
        raw_path = []

        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i+1]
            
            # Robust edge retrieval (handle multigraphs if necessary)
            edge_data = graph.get_edge_data(u, v) or {}
            relation = edge_data.get("relation", "RELATED_TO")

            # Cypher Construction
            match_clauses.append(f"-[:{relation}]->(n{i+1} {{name: '{v}'}})")
            raw_path.append({"source": u, "relation": relation, "target": v})

            # Step Generation
            steps.append({
                "task_id": str(i + 1),
                "agent": "RiskAssessmentAgent",
                "description": f"Verify relationship: {u} --[{relation}]--> {v}",
                "cypher_fragment": f"MATCH (a {{name: '{u}'}})-[:{relation}]->(b {{name: '{v}'}}) RETURN a, b",
                "dependencies": [str(i)] if i > 0 else []  # Implicit sequential dependency
            })

        cypher_query = "MATCH path = " + "".join(match_clauses) + " RETURN path"
        
        return {
            "symbolic_plan": cypher_query,
            "steps": steps,
            "raw_path": raw_path,
            "method": "Graph_Traversal"
        }

    # -------------------------------------------------------------------------
    # 2. NEURAL PARSING (LLM Output Processing)
    # -------------------------------------------------------------------------

    def parse_natural_language_plan(self, text: str) -> Dict[str, Any]:
        """
        Parses LLM output (AWO Phase 1) into a structured plan.
        Includes Topological Sorting to resolve dependencies.
        """
        steps = []
        
        # Regex for "1. [Task]" or "Step 1: [Task]"
        # Captures numbered lists common in CoT (Chain of Thought) outputs
        pattern_std = re.compile(r"^\s*[\*]*(\d+)[\*\.\)]+\s+(.+)$", re.MULTILINE)
        pattern_alt = re.compile(r"^\s*Step\s+(\d+)\s*[:\.]\s+(.+)$", re.MULTILINE | re.IGNORECASE)
        
        matches = pattern_std.findall(text) or pattern_alt.findall(text)

        if not matches:
            logger.warning("No numbered steps found in text. Returning Empty Plan.")
            return {"symbolic_plan": "Manual/Empty", "steps": []}

        for num, desc in matches:
            desc = desc.strip()
            # Naive dependency extraction: looks for "after step X" or "depends on X"
            dep_matches = re.findall(r"(?:step|task)\s+(\d+)", desc.lower())
            dependencies = list(set(dep_matches))

            steps.append({
                "task_id": str(num),
                "agent": "GeneralAgent", # Default, can be refined by NLP classifier
                "description": desc,
                "dependencies": dependencies,
                "cypher_fragment": None
            })

        # CRITICAL: Reorder steps based on dependencies
        sorted_steps = self._topological_sort(steps)
        
        return {
            "symbolic_plan": "Natural Language Parsed",
            "steps": sorted_steps,
            "raw_path": [],
            "method": "LLM_Parse_TopoSort"
        }

    def _topological_sort(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sorts tasks such that if A depends on B, B comes first.
        Uses Depth-First Search (DFS).
        """
        id_to_step = {s['task_id']: s for s in steps}
        adj_list = {s['task_id']: set() for s in steps}

        # Build Graph
        for s in steps:
            for dep in s.get('dependencies', []):
                if dep in id_to_step:
                    adj_list[s['task_id']].add(dep)

        visited = set()
        temp_mark = set() # To detect cycles
        sorted_output = []

        def visit(node_id):
            if node_id in temp_mark:
                # Cycle detected (e.g., A depends on B, B depends on A).
                # We break the cycle by returning immediately (soft fail).
                logger.warning(f"Cycle detected at step {node_id}. Breaking dependency chain.")
                return
            if node_id in visited:
                return

            temp_mark.add(node_id)
            for dep_id in adj_list[node_id]:
                visit(dep_id)
            
            temp_mark.remove(node_id)
            visited.add(node_id)
            sorted_output.append(id_to_step[node_id])

        for s in steps:
            if s['task_id'] not in visited:
                visit(s['task_id'])

        return sorted_output

    # -------------------------------------------------------------------------
    # 3. EXECUTION RUNTIME (LangGraph Integration)
    # -------------------------------------------------------------------------

    def execute_step(self, state: GraphState) -> Dict[str, Any]:
        """
        Worker node for LangGraph. Executes the current step in the plan.
        """
        plan = state.get("plan")
        if not plan:
            return {"assessment": {"error": "Execution halted: No plan provided."}}

        index = state.get("current_task_index", 0)
        steps = plan.get("steps", [])

        if index >= len(steps):
            return {"assessment": {"status": "Complete"}}

        step = steps[index]
        description = step.get("description", "Unknown Task")
        logger.info(f"[Runtime] Executing Step {index+1}/{len(steps)}: {description}")

        # --- REAL AGENT EXECUTION HOOK ---
        # In a production environment, this is where we yield to the Agent
        result_text = f"Simulated Success: {description}"
        
        if self.default_agent:
            # Placeholder for actual async invocation
            # response = await self.default_agent.ainvoke(step)
            # result_text = response.output
            pass

        # Update State
        current_assessment = state.get("assessment") or {}
        content = current_assessment.get("content", "")
        content += f"\n- [Step {index+1}] {result_text}"

        return {
            "assessment": {"content": content},
            "current_task_index": index + 1
        }

    def should_continue(self, state: GraphState) -> str:
        """Conditional logic for LangGraph edges."""
        plan = state.get("plan")
        index = state.get("current_task_index", 0)
        steps = plan.get("steps", [])
        
        if index < len(steps):
            return "continue"
        return "end"

    def to_executable_graph(self, plan_data: Optional[Dict[str, Any]] = None) -> StateGraph:
        """
        Compiles the planner logic into a deployable LangGraph workflow.

        Args:
            plan_data: Optional dictionary containing plan details.
                       Currently used for validation/logging, but can be used
                       to dynamically structure the graph (e.g., adding parallel branches).
        """
        if plan_data:
            logger.info(f"Compiling executable graph for plan with {len(plan_data.get('steps', []))} steps.")

        workflow = StateGraph(GraphState)
        
        workflow.add_node("executor", self.execute_step)
        workflow.set_entry_point("executor")
        
        workflow.add_conditional_edges(
            "executor",
            self.should_continue,
            {
                "continue": "executor",
                "end": END
            }
        )
        
        return workflow.compile()

    # -------------------------------------------------------------------------
    # 4. UTILITIES
    # -------------------------------------------------------------------------

    def validate_plan_logic(self, plan: Dict[str, Any]) -> bool:
        """Basic sanity checks for generated plans."""
        steps = plan.get("steps", [])
        if not steps:
            logger.error("Plan validation failed: Step list is empty.")
            return False
        return True

    def _generate_fallback_plan(self, start_node: str, target_node: str) -> Dict[str, Any]:
        """Safety net: Returns a generic analysis plan if KG traversal fails."""
        steps = [
            {
                "task_id": "1",
                "agent": "RiskAssessmentAgent",
                "description": f"Perform standalone analysis of {start_node}.",
                "dependencies": []
            },
            {
                "task_id": "2",
                "agent": "RiskAssessmentAgent",
                "description": f"Assess theoretical impact on {target_node}.",
                "dependencies": ["1"]
            }
        ]
        return {"symbolic_plan": "Fallback_Generic", "steps": steps, "raw_path": []}
