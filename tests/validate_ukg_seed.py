import json
import os
import sys

def validate_ukg_seed():
    filepath = 'data/v23_ukg_seed.json'
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        sys.exit(1)

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        print(f"Successfully loaded {filepath}")

        ukg = data.get("v23_unified_knowledge_graph", {})
        if not ukg:
            print("Error: Root key 'v23_unified_knowledge_graph' missing.")
            sys.exit(1)

        nodes = ukg.get("nodes", {})
        legal_entities = nodes.get("legal_entities", [])
        financial_instruments = nodes.get("financial_instruments", {})
        esg_profiles = nodes.get("esg_profiles", [])

        print(f"Found {len(legal_entities)} legal entities.")
        print(f"Found {len(financial_instruments.get('loans', []))} loans.")
        print(f"Found {len(financial_instruments.get('securities', []))} securities.")
        print(f"Found {len(esg_profiles)} ESG profiles.")

        # Basic validation of content
        expected_leis = ["8I5DZWZKVSZI1NUHU748", "J3WHBG0MTS7O8ZVMDC91", "INR2EJN1ERAN0W5ZP974"]
        found_leis = [le.get("lei_code") for le in legal_entities]

        for lei in expected_leis:
            if lei not in found_leis:
                print(f"Error: Expected LEI {lei} not found.")
                sys.exit(1)

        print("Validation successful: Data structure matches expectations.")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    validate_ukg_seed()
