import json
import os
import sys

# Add repo root to path
sys.path.append(os.getcwd())

from core.compliance.snc_validators import FinancialHealthCheck

def generate_policy_data():
    manifest = FinancialHealthCheck.get_policy_manifest()

    filepath = "showcase/data/policy_data.json"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"Generated policy data at {filepath}")

if __name__ == "__main__":
    generate_policy_data()
