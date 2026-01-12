# core/engine/neuro_symbolic_planner.py

import logging
import networkx as nx
import re
import asyncio
from typing import List, Dict, Any, Optional, Set
from langgraph.graph import StateGraph, END, START
from enum import Enum

# Internal imports
from core.engine.unified_knowledge_graph import UnifiedKnowledgeGraph
from core.engine.states import GraphState

logger = logging.getLogger(__name__)

class PlannerIntent(Enum):
    DEEP_DIVE = "DEEP_DIVE"
    RISK_ALERT = "RISK_ALERT"
    MARKET_UPDATE = "MARKET_UPDATE"
    GENERAL_QUERY = "GENERAL_QUERY"

class NeuroSymbolicPlanner:
    """
    Implements the Plan-on-Graph (PoG) framework for verifiable, grounded planning.

    Bridge Pattern:
    - Symbolic Side: Uses NetworkX/Neo4j to find valid reasoning paths.
    - Neural Side: Uses LLMs (or semantic emulation) to parse unstructured requests into directed graphs.
    """

    # Bolt Optimization: Pre-compile Regex patterns to avoid re-compilation overhead
    # Pattern for entity extraction: "Analyze [Entity] [Intent]"
    ENTITY_PATTERN = re.compile(
        r"Analyze\s+([A-Za-z0-9\s\.]+?)(?:\s+(?:credit|market|esg|risk|rating).*)?$", re.IGNORECASE
    )

    # Patterns for parsing LLM numbered lists
    # Captures "1. [Task]" or "Step 1: [Task]"
    PATTERN_STD = re.compile(r"^\s*[\*]*(\d+)[\*\.\)]+\s+(.+)$", re.MULTILINE)
    PATTERN_ALT = re.compile(r"^\s*Step\s+(\d+)\s*[:\.]\s+(.+)$", re.MULTILINE | re.IGNORECASE)

    # Pattern for dependency extraction: "after step X" or "depends on X"
    DEP_PATTERN = re.compile(r"(?:step|task)\s+(\d+)", re.IGNORECASE)

    # Patterns for execution step verification
    RELATION_VERIFY_PATTERN = re.compile(r"Verify relationship: (.*?) [-–]")
    ANALYZE_PATTERN = re.compile(r"Analyze (.*?) ", re.IGNORECASE)

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
        1. Classify Intent (Semantic Router)
        2. Extract Entities (Dynamic NER)
        3. Discover Symbolic Path
        4. Fallback to LLM Planning
        """
        logger.info(f"[Planner] Creating plan for request: '{request}'")

        # 1. Semantic Intent Classification
        intent = self._classify_intent_semantic(request)
        logger.info(f"[Planner] Classified Intent: {intent.value}")

        # 2. Dynamic Entity Extraction
        entities = self._extract_entities_dynamic(request)
        start_node = entities.get("primary_entity")

        if start_node:
            # Map Intent to Target Node
            target_node = self._map_intent_to_target(intent, request)

            logger.info(f"[Planner] Extracted Entity: '{start_node}', Inferred Target: '{target_node}'")

            # 3. Symbolic Discovery
            # We try to find a path from Entity -> Risk
            plan = self.discover_plan(start_node, target_node)
            return plan

        # 4. Fallback to NLP Parsing logic
        logger.info("[Planner] Entity extraction failed or no path found. Falling back to NLP parsing.")
        return self.parse_natural_language_plan(request)

    def _classify_intent_semantic(self, request: str) -> PlannerIntent:
        """
        Classifies the user intent using a semantic router.
        (Currently uses weighted keyword scoring as a robust fallback for LLM)
        """
        request_lower = request.lower()

        # Weighted keywords
        scores = {
            PlannerIntent.DEEP_DIVE: 0,
            PlannerIntent.RISK_ALERT: 0,
            PlannerIntent.MARKET_UPDATE: 0
        }

        keywords = {
            PlannerIntent.DEEP_DIVE: ["deep dive", "comprehensive", "full analysis", "valuation", "fundamental", "report"],
            PlannerIntent.RISK_ALERT: ["risk", "alert", "exposure", "default", "crisis", "warning", "downgrade"],
            PlannerIntent.MARKET_UPDATE: ["market", "price", "news", "update", "sentiment", "trend", "moving"]
        }

        for intent, words in keywords.items():
            for word in words:
                if word in request_lower:
                    scores[intent] += 1

        # Determine winner
        best_intent = max(scores, key=scores.get)
        if scores[best_intent] == 0:
            return PlannerIntent.GENERAL_QUERY
        return best_intent

    def _extract_entities_dynamic(self, request: str) -> Dict[str, str]:
        """
        Extracts entities dynamically using regex and heuristic patterns.
        (Placeholder for Named Entity Recognition model)
        """
        # Try specific pattern first
        match = self.ENTITY_PATTERN.search(request)
        if match:
            return {"primary_entity": match.group(1).strip()}

        # Fallback: Look for known tickers or capitalized words in middle of sentence
        # This is a very basic heuristic.
        words = request.split()
        potential_entities = [w for w in words if w.isupper() and len(w) <= 5 and len(w) >= 2]
        if potential_entities:
             # Heuristic: return the first potential ticker
            return {"primary_entity": potential_entities[0].strip(".,!?:")}

        return {}

    def _map_intent_to_target(self, intent: PlannerIntent, request: str) -> str:
        """Maps the classified intent to a target node in the graph."""
        if intent == PlannerIntent.RISK_ALERT:
            if "liquidity" in request.lower(): return "Liquidity_Crisis"
            return "Credit_Default"
        elif intent == PlannerIntent.MARKET_UPDATE:
            return "Market_Crash" # Or 'Market_Volatility' if node exists
        elif intent == PlannerIntent.DEEP_DIVE:
            return "Investment_Decision"

        return "Investment_Decision" # Default

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
            # Check if start node exists (fuzzy match if needed)
            if not graph.has_node(start_node):
                # Try to find case-insensitive match
                nodes = list(graph.nodes())
                start_node_lower = start_node.lower()
                candidates = [n for n in nodes if n.lower() == start_node_lower or start_node_lower in n.lower()]
                if candidates:
                    start_node = candidates[0]
                    logger.info(f"[PoG] Fuzzy matched '{start_node_lower}' to '{start_node}'")
                else:
                    raise nx.NodeNotFound(f"Node {start_node} not found")

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

            # RAG-Guided Enhancement: Inject semantic context if available
            # In a full vector system, we would query the index here for "Why is {u} related to {v}?"
            # Simulated RAG Step:
            rag_context = f"Context: Investigating semantic link between {u} and {v}."

            # Step Generation
            steps.append({
                "task_id": str(i + 1),
                "agent": "RiskAssessmentAgent",
                "description": f"Verify relationship: {u} --[{relation}]--> {v}. {rag_context}",
                "cypher_fragment": f"MATCH (a {{name: '{u}'}})-[:{relation}]->(b {{name: '{v}'}}) RETURN a, b",
                "dependencies": [str(i)] if i > 0 else []  # Implicit sequential dependency
            })

        cypher_query = "MATCH path = " + "".join(match_clauses) + " RETURN path"

        return {
            "symbolic_plan": cypher_query,
            "steps": steps,
            "raw_path": raw_path,
            "method": "RAG_Guided_Traversal"
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

        # Uses pre-compiled regex patterns
        matches = self.PATTERN_STD.findall(text) or self.PATTERN_ALT.findall(text)

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
            # Bolt Optimization: Avoids desc.lower() by using re.IGNORECASE in DEP_PATTERN
            dep_matches = self.DEP_PATTERN.findall(desc)
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
                # Bolt Optimization: Removed unnecessary list copy
                for dep_id in adj_list[node_id]:
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
                match = self.RELATION_VERIFY_PATTERN.search(description)
                if match:
                    target_entity = match.group(1).strip()
                else:
                    # Regex 2: General analysis pattern (Analyze X...)
                    match_req = self.ANALYZE_PATTERN.search(description)
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
        """
        Safety net: Returns a generic analysis plan if KG traversal fails.
        Now uses 'Vector Anchoring' simulation to find relevant sub-topics.
        """
        # Simulated Vector Search for related topics
        topics = ["Financial Health", "Market Sentiment", "Regulatory Risk"]

        steps = [
            {
                "task_id": "1",
                "agent": "RiskAssessmentAgent",
                "description": f"Perform deep dive analysis of {start_node} focusing on {topics[0]}.",
                "dependencies": []
            },
            {
                "task_id": "2",
                "agent": "RiskAssessmentAgent",
                "description": f"Analyze {start_node} regarding {topics[1]}.",
                "dependencies": ["1"]
            },
            {
                "task_id": "3",
                "agent": "RiskAssessmentAgent",
                "description": f"Evaluate {start_node} for {target_node} exposure.",
                "dependencies": ["1", "2"]
            }
        ]
        return {"symbolic_plan": "RAG_Fallback_Anchored", "steps": steps, "raw_path": []}
