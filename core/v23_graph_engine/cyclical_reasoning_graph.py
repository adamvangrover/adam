import logging
from typing import Literal

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from core.v23_graph_engine.states import GraphState
from core.v23_graph_engine.neuro_symbolic_planner import NeuroSymbolicPlanner
from core.v23_graph_engine.self_reflection_agent import SelfReflectionAgent

logger = logging.getLogger(__name__)

class CyclicalReasoningGraph:
    """
    Implements the 'Reasoning' engine.
    Nodes: Plan -> Draft -> Critique -> Refine -> End
    """

    def __init__(self):
        self.planner = NeuroSymbolicPlanner()
        self.critic = SelfReflectionAgent()
        self.workflow = None

        if LANGGRAPH_AVAILABLE:
            self._build_graph()
        else:
            logger.warning("LangGraph not available. Graph will not be compiled.")

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        # Define Nodes
        workflow.add_node("planner", self.plan_node)
        workflow.add_node("drafter", self.draft_node)
        workflow.add_node("critic", self.critique_node)
        workflow.add_node("refiner", self.refine_node)

        # Define Edges
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "drafter")
        workflow.add_edge("drafter", "critic")

        # Conditional Edge
        workflow.add_conditional_edges(
            "critic",
            self.should_continue,
            {
                "pass": END,
                "fail": "refiner"
            }
        )
        workflow.add_edge("refiner", "critic") # Loop back to critique

        self.workflow = workflow.compile()

    # --- Node Functions ---

    def plan_node(self, state: GraphState):
        logger.info("Executing Node: Planner")
        query = state["request"].query if state.get("request") else "Default Query"
        plan = self.planner.generate_plan(query)
        return {"plan": plan}

    def draft_node(self, state: GraphState):
        logger.info("Executing Node: Drafter")
        # Simulate drafting based on plan
        plan = state.get("plan")
        draft = f"Initial Draft based on plan {plan.plan_id if plan else 'None'}.\n"
        draft += "Sections: Overview, Financials.\n"
        # Deliberately make it fail the first check if we want to test the loop
        if state.get("revision_count", 0) == 0:
            draft += "Note: Preliminary data."
        else:
            draft += "Risk Analysis: Comprehensive review of liquidity and credit. Risk is Moderate."

        return {"draft": draft}

    def critique_node(self, state: GraphState):
        logger.info("Executing Node: Critic")
        result = self.critic.critique(state)
        return {"critique": result}

    def refine_node(self, state: GraphState):
        logger.info("Executing Node: Refiner")
        current_revision = state.get("revision_count", 0)
        feedback = state["critique"].feedback

        # Refine logic
        new_draft = state["draft"] + f"\n[Revision {current_revision+1}] Addressing feedback: {feedback}"

        return {
            "draft": new_draft,
            "revision_count": current_revision + 1
        }

    # --- Conditional Logic ---

    def should_continue(self, state: GraphState) -> Literal["pass", "fail"]:
        critique = state.get("critique")
        revision_count = state.get("revision_count", 0)

        if critique and critique.passed:
            return "pass"

        if revision_count >= 3: # Max loops
            logger.warning("Max revisions reached. Forcing exit.")
            return "pass"

        return "fail"

    def invoke(self, inputs):
        if self.workflow:
            return self.workflow.invoke(inputs)
        else:
            logger.error("Workflow not compiled (LangGraph missing).")
            return inputs
