# core/v23_graph_engine/snc_utils.py

"""
Agent Notes (Meta-Commentary):
Helper module for Shared National Credit (SNC) analysis logic.
Encapsulates pure functions for financial calculations, vote counting,
and regulatory classification rules.
"""

from typing import Dict, Any, List

def calculate_leverage(debt: float, ebitda: float) -> float:
    """Calculates Debt/EBITDA leverage ratio."""
    if ebitda == 0:
        return 999.9 # Avoid div by zero
    return debt / ebitda

def check_covenant_compliance(current_metric: float, covenant_limit: float, metric_type: str = "max") -> bool:
    """
    Checks if a metric complies with a covenant.
    metric_type: 'max' (e.g. Leverage < 5.0) or 'min' (e.g. Coverage > 1.2)
    """
    if metric_type == "max":
        return current_metric <= covenant_limit
    elif metric_type == "min":
        return current_metric >= covenant_limit
    return False

def determine_vote_outcome(banks: List[Dict[str, Any]], required_threshold: float = 0.51) -> str:
    """
    Determines if a syndicate vote passes based on pro-rata shares.
    banks: List of dicts with 'share' (0.0 to 1.0) and 'vote' ('Yes'/'No').
    """
    yes_votes = sum(b['share'] for b in banks if b.get('vote') == 'Yes')
    return "Pass" if yes_votes >= required_threshold else "Fail"

def map_financials_to_rating(leverage: float, liquidity: float, debt: float) -> str:
    """
    Heuristic rule-based mapping for initial regulatory rating draft.
    Note: Real rating requires qualitative analysis; this is a baseline.
    """
    if leverage > 6.0 and liquidity < (debt * 0.05):
        return "Substandard"
    elif leverage > 5.0 or liquidity < (debt * 0.10):
        return "Special Mention"
    else:
        return "Pass"

def analyze_syndicate_structure(syndicate_data: Dict[str, Any]) -> str:
    """
    Analyzes the syndicate structure for concentration risk.
    """
    banks = syndicate_data.get("banks", [])
    lead_share = next((b['share'] for b in banks if b.get('role') == 'Lead'), 0.0)

    analysis = f"Syndicate Size: {len(banks)} institutions.\n"
    if lead_share > 0.5:
        analysis += f"Concentration Risk: Lead bank holds {lead_share:.1%} of the facility, which is high.\n"
    elif lead_share < 0.1:
        analysis += f"Leadership Risk: Lead bank holds only {lead_share:.1%}, potentially reducing 'skin in the game'.\n"
    else:
        analysis += f"Structure: Balanced. Lead bank holds {lead_share:.1%}.\n"

    return analysis
