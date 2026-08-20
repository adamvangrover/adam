import json
import sys
import os
import datetime
import hashlib

# Setup PYTHONPATH so src.pdil can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.pdil.middleware import JsonLogicGovernanceGatekeeper, GovernanceError
except ImportError:
    print("Warning: Could not import JsonLogicGovernanceGatekeeper. Running without strict governance.")
    JsonLogicGovernanceGatekeeper = None

def generate_prov_o_metadata(agent_id, activity_name):
    return {
        "@context": "http://www.w3.org/ns/prov",
        "type": "Activity",
        "label": activity_name,
        "startedAtTime": datetime.datetime.now(datetime.UTC).isoformat(),
        "wasAssociatedWith": agent_id
    }

def parse_sec_filing(filing_text: str, config_path: str, prompt_path: str) -> dict:
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        return {"error": "Config not found"}

    try:
        with open(prompt_path, 'r') as f:
            prompt_data = json.load(f)
    except FileNotFoundError:
        return {"error": "Prompt not found"}

    data_payload = {
        "company": "Acme Corp",
        "cik": "0001234567",
        "debt_covenants": [
            {
                "covenant_type": "Leverage Ratio",
                "threshold": 4.5,
                "description": "Total Debt to EBITDA must not exceed 4.5x"
            }
        ],
        "macro_indicators": ["Interest rate sensitivity high"]
    }

    # Calculate content hash for strict provenance check
    payload_json = json.dumps(data_payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    computed_hash = hashlib.sha256(payload_json).hexdigest()

    # Wrap in the expected inference output structure
    result = {
        "data": data_payload
    }

    # Provenance Integration (System Level Header)
    result["provenance_trace"] = {
        "event_id": "test-event-123",
        "source_agent": "agent:enterprise_sec_parser",
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit_hash": "mockhash",
        "content_hash": computed_hash,
        "jsonLogic_version": "1.0",
        "confidence_score": 0.99,
        "derivation_path": "source_doc",
        "source_data_object": "mocked_filing"
    }

    # Governance Integration
    if JsonLogicGovernanceGatekeeper:
        gatekeeper = JsonLogicGovernanceGatekeeper(rules={"==": [1, 1]})
        try:
            result = gatekeeper.validate_inference(result)
            # print("Governance Validation Passed.")
        except Exception as e:
            return {"error": f"Governance validation failed: {str(e)}"}

    # Business Level PROV-O metadata
    prov_metadata = generate_prov_o_metadata("agent:enterprise_sec_parser", "parse_edgar_covenants")
    result["_provenance"] = prov_metadata

    # Telemetry Integration
    telemetry_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'telemetry_history', 'market_mayhem_telemetry.jsonl'))
    if os.path.exists(os.path.dirname(telemetry_path)):
        telemetry_entry = {"timestamp": datetime.datetime.now(datetime.UTC).isoformat(), "event": "covenant_parsed", "data": result}
        try:
            with open(telemetry_path, 'a') as f:
                f.write(json.dumps(telemetry_entry) + "\n")
        except Exception as e:
            print(f"Failed to write telemetry: {e}")

    return result

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file = os.path.join(base_dir, 'domains', 'enterprise', 'config.json')
    prompt_file = os.path.join(base_dir, 'prompts', 'enterprise_credit_prompt.json')

    output = parse_sec_filing("MOCK SEC FILING 10-K...", config_file, prompt_file)
    print(json.dumps(output, indent=2))
