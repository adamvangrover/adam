
import logging
import random
from typing import Dict, Any, Literal

from langgraph.graph import StateGraph, END
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
    Enhancement: Applies Counterfactual Logic ("What If?") to test robustness.
    """
    desc = state["current_scenario_description"]

    # Mock simulation logic
    base_impact = random.uniform(3.0, 8.0)
    escalation_bonus = state["iteration_count"] * 1.5

    # Counterfactual Logic Simulation
    # We ask: "What if the entity has mitigation X?"
    mitigation_factor = 0.0
    cf_note = "Base Assumption: No Mitigation."

    if "Cyber" in desc:
        # CF: What if they have air-gapped backups?
        if random.random() > 0.5:
            mitigation_factor = 2.0
            cf_note = "Counterfactual: Entity has air-gapped backups (Impact Reduced)."
    elif "Macro" in desc:
        # CF: What if they have interest rate swaps?
        if random.random() > 0.5:
            mitigation_factor = 1.5
            cf_note = "Counterfactual: Interest Rate Swaps active (Impact Reduced)."

    total_impact = min(10.0, max(0.0, base_impact + escalation_bonus - mitigation_factor))

    logger.info(f"Simulated impact for '{desc}': {total_impact:.2f}/10.0. {cf_note}")

    return {
        "simulated_impact_score": total_impact,
        "human_readable_status": f"Simulating impact... Score: {total_impact:.1f} ({cf_note})"
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
        notes.append(f"Pass: Impact {impact:.1f} meets threshold {threshold}. Valid Stress Test.")
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
