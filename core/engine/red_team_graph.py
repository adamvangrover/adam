
import logging
import random
from typing import Dict, Any, Literal

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    class StateGraph:
         def __init__(self, *args, **kwargs): pass
         def add_node(self, *args, **kwargs): pass
         def add_edge(self, *args, **kwargs): pass
         def set_entry_point(self, *args, **kwargs): pass
         def add_conditional_edges(self, *args, **kwargs): pass
         def compile(self, *args, **kwargs): return None
    END = "END"
    logger.warning("LangGraph not installed. Graphs will be disabled.")
from core.engine.states import RedTeamState

logger = logging.getLogger(__name__)

# --- Nodes ---


def generate_attack_node(state: RedTeamState) -> Dict[str, Any]:
    """
    Node: Generates or refines an adversarial scenario.
    """
    target = state["target_entity"]
    scenario_type = state["scenario_type"]
    iteration = state["iteration_count"]

    # Mock generation logic (In real life, this is an LLM call)
    base_scenarios = {
        "Cyber": f"Distributed Denial of Service (DDoS) attack on {target}'s primary transaction servers.",
        "Macro": f"Unexpected 50bps interest rate hike by the Fed affecting {target}'s debt servicing.",
        "Regulatory": f"New GDPR investigation opened into {target}'s data handling practices."
    }

    current_desc = base_scenarios.get(scenario_type, f"Generic stress test on {target}")

    if iteration > 0:
        current_desc += f" (Escalation Level {iteration}: Attack vector widened to include supply chain partners.)"

    logger.info(f"Red Team generating scenario: {current_desc}")

    return {
        "current_scenario_description": current_desc,
        "iteration_count": iteration + 1,
        "human_readable_status": f"Drafting scenario: {scenario_type} (Iter {iteration})"
    }


def simulate_impact_node(state: RedTeamState) -> Dict[str, Any]:
    """
    Node: Simulates the impact of the scenario on the target.
    """
    desc = state["current_scenario_description"]

    # Mock simulation logic (In real life, this queries the KG or a quantitative model)
    # We assign a random impact score for demo purposes, biased by iteration count
    base_impact = random.uniform(2.0, 6.0)
    escalation_bonus = state["iteration_count"] * 1.5
    total_impact = min(10.0, base_impact + escalation_bonus)

    logger.info(f"Simulated impact for '{desc}': {total_impact:.2f}/10.0")

    return {
        "simulated_impact_score": total_impact,
        "human_readable_status": f"Simulating impact... Score: {total_impact:.1f}"
    }


def critique_node(state: RedTeamState) -> Dict[str, Any]:
    """
    Node: Critiques whether the scenario is severe enough to be a valid stress test.
    """
    impact = state["simulated_impact_score"]
    threshold = state["severity_threshold"]

    is_severe = impact >= threshold

    notes = state["critique_notes"]
    if is_severe:
        notes.append(f"Pass: Impact {impact:.1f} meets threshold {threshold}.")
    else:
        notes.append(f"Fail: Impact {impact:.1f} below threshold {threshold}. Needs escalation.")

    return {
        "is_sufficiently_severe": is_severe,
        "critique_notes": notes,
        "human_readable_status": "Critiquing scenario severity..."
    }

# --- Conditional Logic ---


def should_continue(state: RedTeamState) -> Literal["escalate", "finalize"]:
    if state["is_sufficiently_severe"]:
        return "finalize"

    if state["iteration_count"] >= 3:
        # Prevent infinite loops
        return "finalize"

    return "escalate"


def finalize_node(state: RedTeamState) -> Dict[str, Any]:
    return {
        "human_readable_status": "Red Team scenario finalized."
    }

# --- Graph Construction ---


def build_red_team_graph():
    if not HAS_LANGGRAPH:
        return None

    workflow = StateGraph(RedTeamState)

    workflow.add_node("generate_attack", generate_attack_node)
    workflow.add_node("simulate_impact", simulate_impact_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("generate_attack")

    workflow.add_edge("generate_attack", "simulate_impact")
    workflow.add_edge("simulate_impact", "critique")

    workflow.add_conditional_edges(
        "critique",
        should_continue,
        {
            "escalate": "generate_attack",
            "finalize": "finalize"
        }
    )

    workflow.add_edge("finalize", END)

    return workflow.compile()


red_team_app = build_red_team_graph()
