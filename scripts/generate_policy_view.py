import json
import os
import sys
# Ensure repo root
sys.path.append(os.getcwd())

from core.compliance.snc_validators import FinancialHealthCheck

OUTPUT_FILE = "showcase/data/policy_data.json"

def generate_policy_data():
    # Extract SNC Limits from ClassVar
    snc_rules = []

    # 1. Base Thresholds
    snc_rules.append({
        "name": "Max Leverage (General)",
        "value": f"{FinancialHealthCheck.MAX_DEBT_TO_EQUITY}x",
        "description": "Maximum allowable Debt-to-Equity ratio for general borrowers."
    })
    snc_rules.append({
        "name": "Min Interest Coverage",
        "value": f"{FinancialHealthCheck.MIN_INTEREST_COVERAGE}x",
        "description": "Minimum EBIT / Interest Expense ratio."
    })

    # 2. Sector Specifics (Iterate over dict)
    for sector, limits in FinancialHealthCheck.SECTOR_LIMITS.items():
        for limit_name, val in limits.items():
            readable_name = limit_name.replace("_", " ").title()
            snc_rules.append({
                "name": f"{readable_name} ({sector.value})",
                "value": str(val),
                "description": f"Sector-specific override for {sector.value}."
            })

    # Mock Governance Rules (normally read from yaml)
    governance_rules = [
        "No unreviewed code merges to `main`.",
        "All agents must log decisions to `audit_trails`.",
        "HFT algorithms strictly limited to $1M daily volume in simulation.",
        "Model weights must be versioned with SHA-256 hashes."
    ]

    data = {
        "snc_rules": snc_rules,
        "governance_rules": governance_rules
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Policy data generated at {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_policy_data()
