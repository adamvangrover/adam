import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.compliance.snc_validators import FinancialHealthCheck, SectorType

def generate_policy_data():
    data = {
        "global_limits": {
            "MAX_DEBT_TO_EQUITY": FinancialHealthCheck.MAX_DEBT_TO_EQUITY,
            "MIN_PROFIT_MARGIN": FinancialHealthCheck.MIN_PROFIT_MARGIN,
            "MIN_CURRENT_RATIO": FinancialHealthCheck.MIN_CURRENT_RATIO,
            "MIN_INTEREST_COVERAGE": FinancialHealthCheck.MIN_INTEREST_COVERAGE
        },
        "sector_overrides": {}
    }

    # Iterate over SectorType enum
    for sector in SectorType:
        overrides = FinancialHealthCheck.SECTOR_LIMITS.get(sector, {})
        # Ensure the overrides are JSON serializable (they are floats, so yes)

        # We want to present the *effective* limits for each sector to the UI
        # So let's merge global and override
        effective_limits = {
            "MAX_DEBT_TO_EQUITY": overrides.get("MAX_DEBT_TO_EQUITY", FinancialHealthCheck.MAX_DEBT_TO_EQUITY),
            "MIN_PROFIT_MARGIN": overrides.get("MIN_PROFIT_MARGIN", FinancialHealthCheck.MIN_PROFIT_MARGIN),
            "MIN_CURRENT_RATIO": overrides.get("MIN_CURRENT_RATIO", FinancialHealthCheck.MIN_CURRENT_RATIO),
            "MIN_INTEREST_COVERAGE": overrides.get("MIN_INTEREST_COVERAGE", FinancialHealthCheck.MIN_INTEREST_COVERAGE)
        }

        data["sector_overrides"][sector.value] = {
            "overrides": overrides,
            "effective_limits": effective_limits
        }

    output_dir = "showcase/data"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "policy_data.json")
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Policy data generated at {output_file}")

if __name__ == "__main__":
    generate_policy_data()
