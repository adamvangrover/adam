# core/engine/regulatory_compliance_graph.py

"""
Agent Notes (Meta-Commentary):
This module implements the Regulatory Compliance Graph.
It automates the process of checking a financial entity against complex regulatory frameworks
(e.g., Basel III, Dodd-Frank, KYC/AML).
It uses a cyclical approach to ensure no violation is missed and interpretations are double-checked.
"""

import logging
from typing import Literal, Dict, Any

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
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
    START = "START"
    class MemorySaver: pass
    logger.warning("LangGraph not installed. Graphs will be disabled.")

from core.engine.states import ComplianceState

# --- Mock Utilities ---


def mock_get_regulations(jurisdiction: str) -> list[str]:
    regs = ["KYC - Know Your Customer", "AML - Anti-Money Laundering"]
    if jurisdiction.upper() == "US":
        regs.extend(["Dodd-Frank", "CCAR"])
    elif jurisdiction.upper() == "EU":
        regs.extend(["Basel III", "GDPR", "MiFID II"])
    return regs


def mock_check_violation_logic(entity: str, reg: str) -> bool:
    # Randomly simulate a violation for "Crypto" entities
    if "Crypto" in entity and "AML" in reg:
        return True
    return False

# --- Nodes ---


def identify_jurisdiction_node(state: ComplianceState) -> Dict[str, Any]:
    print("--- Node: Identify Jurisdiction ---")
    # In a real system, we'd infer this from the entity metadata
    jur = state["jurisdiction"] or "US"
    return {
        "jurisdiction": jur,
        "human_readable_status": f"identified jurisdiction: {jur}"
    }


def fetch_regulations_node(state: ComplianceState) -> Dict[str, Any]:
    print("--- Node: Fetch Regulations ---")
    regs = mock_get_regulations(state["jurisdiction"])
    return {
        "applicable_regulations": regs,
        "human_readable_status": f"Fetched {len(regs)} applicable regulations."
    }


def check_compliance_node(state: ComplianceState) -> Dict[str, Any]:
    print("--- Node: Check Compliance ---")
    violations = []
    risk = "LOW"

    for reg in state["applicable_regulations"]:
        if mock_check_violation_logic(state["entity_id"], reg):
            violations.append(f"Potential violation of {reg}")

    if violations:
        risk = "HIGH" if len(violations) > 1 else "MEDIUM"

    return {
        "potential_violations": violations,
        "risk_level": risk,
        "human_readable_status": f"Compliance check complete. Risk Level: {risk}"
    }


def generate_report_node(state: ComplianceState) -> Dict[str, Any]:
    print("--- Node: Generate Compliance Report ---")
    report = f"Compliance Report for {state['entity_id']}\n"
    report += f"Jurisdiction: {state['jurisdiction']}\n"
    report += f"Risk Level: {state['risk_level']}\n"

    if state["potential_violations"]:
        report += "VIOLATIONS DETECTED:\n"
        for v in state["potential_violations"]:
            report += f"- {v}\n"
    else:
        report += "No violations detected.\n"

    return {
        "final_report": report,
        "human_readable_status": "Generated final report."
    }


def critique_compliance_node(state: ComplianceState) -> Dict[str, Any]:
    print("--- Node: Critique Compliance ---")
    risk = state["risk_level"]
    iteration = state["iteration_count"]

    critique_notes = []
    needs_revision = False

    if risk == "LOW" and "Crypto" in state["entity_id"]:
        critique_notes.append("Entity is in high-risk sector (Crypto). Verify KYC specifically.")
        needs_revision = True

    if iteration < 1:
        critique_notes.append("Cross-reference with OFAC sanctions list (simulated).")
        needs_revision = True

    return {
        "critique_notes": critique_notes,
        "needs_revision": needs_revision,
        "iteration_count": iteration + 1,
        "human_readable_status": "Critiqued compliance findings."
    }


def revise_compliance_node(state: ComplianceState) -> Dict[str, Any]:
    print("--- Node: Revise Compliance ---")
    report = state["final_report"]
    notes = state["critique_notes"]

    new_report = report + "\n[Compliance Officer Notes]\n"
    for note in notes:
        new_report += f"- Action Item: {note}\n"

    return {
        "final_report": new_report,
        "needs_revision": False,
        "human_readable_status": "Revised compliance report."
    }

# --- Conditional Logic ---


def should_continue_compliance(state: ComplianceState) -> Literal["revise_compliance", "END"]:
    if state["needs_revision"] and state["iteration_count"] < 3:
        return "revise_compliance"
    return "END"

# --- Graph Construction ---


def build_compliance_graph():
    if not HAS_LANGGRAPH:
        return None

    workflow = StateGraph(ComplianceState)

    workflow.add_node("identify_jurisdiction", identify_jurisdiction_node)
    workflow.add_node("fetch_regulations", fetch_regulations_node)
    workflow.add_node("check_compliance", check_compliance_node)
    workflow.add_node("generate_report", generate_report_node)
    workflow.add_node("critique_compliance", critique_compliance_node)
    workflow.add_node("revise_compliance", revise_compliance_node)

    workflow.add_edge(START, "identify_jurisdiction")
    workflow.add_edge("identify_jurisdiction", "fetch_regulations")
    workflow.add_edge("fetch_regulations", "check_compliance")
    workflow.add_edge("check_compliance", "generate_report")
    workflow.add_edge("generate_report", "critique_compliance")

    workflow.add_conditional_edges(
        "critique_compliance",
        should_continue_compliance,
        {
            "revise_compliance": "revise_compliance",
            "END": END
        }
    )

    workflow.add_edge("revise_compliance", "critique_compliance")

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


compliance_graph_app = build_compliance_graph()
