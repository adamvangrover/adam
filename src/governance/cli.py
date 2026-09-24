import argparse
import sys
import json
from src.governance.gatekeeper import GovernanceGatekeeper

def main():
    parser = argparse.ArgumentParser(description="Adam Security & Governance CLI")
    parser.add_argument("--validate", type=str, help="Path to inference JSON to validate")
    parser.add_argument("--schema", type=str, help="Path to JSON schema constraints")
    
    args = parser.parse_args()
    
    if args.validate:
        with open(args.validate, "r") as f:
            inference = json.load(f)
            
        schema = {"type": "object"}
        if args.schema:
             with open(args.schema, "r") as f:
                 schema = json.load(f)
                 
        gatekeeper = GovernanceGatekeeper(schema=schema)
        try:
            gatekeeper.validate_inference(inference)
            print("Validation successful.")
        except Exception as e:
            print(f"Validation failed: {e}")
            sys.exit(1)
            
if __name__ == "__main__":
    main()
