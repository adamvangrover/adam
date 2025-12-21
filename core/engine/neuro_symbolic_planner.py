# core/engine/neuro_symbolic_planner.py

import logging
import networkx as nx
import re
import asyncio
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

    def create_plan(self, request: str) -> Dict[str, Any]:
        """
        Orchestrates the planning process (High-Level API):
        1. Parse Request
        2. Extract Entities
        3. Discover Symbolic Path
        4. Fallback to LLM Planning
        """
        logger.info(f"[Planner] Creating plan for request: '{request}'")

        # 1. Entity Extraction (Heuristic/Regex)
        # Try to find "Analyze [Entity] [Intent]" or just "Analyze [Entity]"
        # Regex: Analyze (Group 1: Entity) (Optional Group 2: Intent)
        entity_match = re.search(
            r"Analyze\s+([A-Za-z0-9\s\.]+?)(?:\s+(?:credit|market|esg|risk|rating).*)?$", request, re.IGNORECASE)

        if entity_match:
            start_node = entity_match.group(1).strip()
            # Clean up common suffixes for graph matching if needed,
            # but keeping it raw is often safer for fuzzy matching in KG

            # Intent mapping (simple keyword search in the whole request)
            request_lower = request.lower()
            target_node = "Investment_Decision"  # Default generic target
            if "credit" in request_lower:
                target_node = "Credit_Default"
            elif "market" in request_lower:
                target_node = "Market_Crash"
            elif "esg" in request_lower:
                target_node = "ESG_Controversy"
            elif "liquidity" in request_lower:
                target_node = "Liquidity_Crisis"

            logger.info(f"[Planner] Extracted Entity: '{start_node}', Inferred Target: '{target_node}'")

            # 2. Symbolic Discovery
            # We try to find a path from Entity -> Risk
            plan = self.discover_plan(start_node, target_node)

            # If discovery found a path (even if it's a fallback), use it.
            # But if it returned a Generic Fallback because of no path, we might want to try NLP parsing
            # if we had a real LLM connected. Here we stick to the plan returned.
            return plan

        # 3. Fallback to NLP Parsing logic (simulated here via simple regex on numbered lists)
        logger.info("[Planner] Entity extraction failed. Falling back to NLP parsing.")
        return self.parse_natural_language_plan(request)

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
            # If no steps found, create a single generic step
            return {
                "symbolic_plan": "Manual/One-Shot",
                "steps": [{
                    "task_id": "1",
                    "agent": "GeneralAgent",
                    "description": text.strip() or "Execute Request",
                    "dependencies": []
                }]
            }

        for num, desc in matches:
            desc = desc.strip()
            # Naive dependency extraction: looks for "after step X" or "depends on X"
            dep_matches = re.findall(r"(?:step|task)\s+(\d+)", desc.lower())
            dependencies = list(set(dep_matches))

            steps.append({
                "task_id": str(num),
                "agent": "GeneralAgent",  # Default, can be refined by NLP classifier
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
        Uses Depth-First Search (DFS) with Robust Cycle Breaking.
        """
        id_to_step = {s['task_id']: s for s in steps}
        adj_list = {s['task_id']: set() for s in steps}

        # Build Graph
        for s in steps:
            for dep in s.get('dependencies', []):
                if dep in id_to_step:
                    adj_list[s['task_id']].add(dep)
                else:
                    logger.debug(f"Task {s['task_id']} depends on missing task {dep}. Ignoring dependency.")

        visited = set()
        temp_mark = set()  # To detect cycles
        sorted_output = []

        def visit(node_id):
            if node_id in temp_mark:
                # Cycle detected (e.g., A depends on B, B depends on A).
                # We break the cycle by returning immediately (soft fail).
                # This treats the current edge as non-existent for the purpose of sorting.
                logger.warning(f"Cycle detected at step {node_id}. Breaking dependency chain to ensure execution.")
                return
            if node_id in visited:
                return

            temp_mark.add(node_id)
            try:
                # Iterate over a copy to allow modification if needed (though we don't modify here)
                for dep_id in list(adj_list[node_id]):
                    visit(dep_id)
            finally:
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

    async def execute_step(self, state: GraphState) -> Dict[str, Any]:
        """
        Worker node for LangGraph. Executes the current step in the plan.
        Merged Logic: Prioritizes Async execution with fallback.
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
        agent_name = step.get("agent")

        logger.info(f"[Planner] Executing Step {index + 1}/{len(steps)}: {description}")

        # Default result text
        result_text = f"Executed: {description}"

        if self.default_agent and (agent_name == "RiskAssessmentAgent" or agent_name == "GeneralAgent"):
            try:
                # Attempt to extract entity from description (simple heuristic for graph traversal steps)
                # e.g. "Verify relationship: Apple Inc. -[SUPPLIER]-> Foxconn"
                target_entity = "Unknown"

                # Regex 1: Relationship verification pattern
                match = re.search(r"Verify relationship: (.*?) [-–]", description)
                if match:
                    target_entity = match.group(1).strip()
                else:
                    # Regex 2: General analysis pattern (Analyze X...)
                    match_req = re.search(r"Analyze (.*?) ", description, re.IGNORECASE)
                    if match_req:
                        target_entity = match_req.group(1).strip()
                    else:
                        # Fallback: Check the state for a global target
                        target_entity = state.get("target", state.get("ticker", "Unknown"))

                logger.info(f"[Planner] Delegating to RiskAssessmentAgent for {target_entity}...")

                # Execute Async
                # We mock the payload as RiskAssessmentAgent expects specific keys.
                # In a full system, this would fetch real data from a Data Retrieval Agent.
                target_data = {
                    "company_name": target_entity,
                    "financial_data": {"industry": "Technology"},  # Contextual mock
                    "market_data": {}
                }

                # Check if execute is async
                if asyncio.iscoroutinefunction(self.default_agent.execute):
                    agent_result = await self.default_agent.execute(
                        target_data=target_data,
                        context={"user_intent": description}
                    )
                else:
                    # Sync fallback
                    agent_result = self.default_agent.execute(
                        target_data=target_data,
                        context={"user_intent": description}
                    )

                score = agent_result.get("overall_risk_score", "N/A")
                factors = list(agent_result.get("risk_factors", {}).keys())
                result_text = f"Risk Analysis for {target_entity}: Score {score}. Factors Analyzed: {factors}"

            except Exception as e:
                logger.error(f"[Planner] Agent execution failed: {e}", exc_info=True)
                result_text += f" (Agent Error: {e})"
        else:
            # Simulated execution for other agents or if agent not loaded
            result_text = f"Simulated Success: {description}"

        # Update State
        current_assessment = state.get("assessment") or {}
        # Ensure content is a string
        content = current_assessment.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        content += f"\n- [Step {index+1}] {result_text}"

        # Return minimal update
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
