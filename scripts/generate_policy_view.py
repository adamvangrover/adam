import json
import os
import sys

# Ensure repo root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.compliance.snc_validators import FinancialHealthCheck, SectorType

OUTPUT_FILE = "showcase/data/policy_data.json"

def generate_policy_data():
    """
    Generates policy data for the showcase UI, combining structured limits
    and human-readable rule lists.
    """
    
    # 1. Structured Data (Machine Readable)
    global_limits = {
        "MAX_DEBT_TO_EQUITY": FinancialHealthCheck.MAX_DEBT_TO_EQUITY,
        "MIN_PROFIT_MARGIN": FinancialHealthCheck.MIN_PROFIT_MARGIN,
        "MIN_CURRENT_RATIO": FinancialHealthCheck.MIN_CURRENT_RATIO,
        "MIN_INTEREST_COVERAGE": FinancialHealthCheck.MIN_INTEREST_COVERAGE
    }
    
    sector_data = {}
    
    # Iterate over SectorType enum to ensure comprehensive coverage
    for sector in SectorType:
        overrides = FinancialHealthCheck.SECTOR_LIMITS.get(sector, {})
        
        # Merge global and overrides to get effective limits
        effective_limits = global_limits.copy()
        effective_limits.update(overrides)

        sector_data[sector.value] = {
            "overrides": overrides,
            "effective_limits": effective_limits
        }

    # 2. Flattened View (UI/Table Friendly)
    # This matches the 'snc_rules' format expected by some UI components
    snc_rules_view = []

    # Base Thresholds
    snc_rules_view.append({
        "name": "Max Leverage (General)",
        "value": f"{FinancialHealthCheck.MAX_DEBT_TO_EQUITY}x",
        "description": "Maximum allowable Debt-to-Equity ratio for general borrowers."
    })
    snc_rules_view.append({
        "name": "Min Interest Coverage",
        "value": f"{FinancialHealthCheck.MIN_INTEREST_COVERAGE}x",
        "description": "Minimum EBIT / Interest Expense ratio."
    })
    
    # Sector Specifics for View
    for sector, limits in FinancialHealthCheck.SECTOR_LIMITS.items():
        for limit_name, val in limits.items():
            readable_name = limit_name.replace("_", " ").title()
            snc_rules_view.append({
                "name": f"{readable_name} ({sector.value})",
                "value": str(val),
                "description": f"Sector-specific override for {sector.value}."
            })

    # 3. Governance Rules (Mock/Static)
    # Normally read from a yaml config, but hardcoded here for showcase generation
    governance_rules = [
        "No unreviewed code merges to `main`.",
        "All agents must log decisions to `audit_trails`.",
        "HFT algorithms strictly limited to $1M daily volume in simulation.",
        "Model weights must be versioned with SHA-256 hashes."
    ]

    # Combine all data
    data = {
        "global_limits": global_limits,
        "sector_overrides": sector_data,
        "snc_rules": snc_rules_view,     # Kept for backward compatibility/UI tables
        "governance_rules": governance_rules
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Policy data generated at {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_policy_data()