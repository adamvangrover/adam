import json
import sys
import os

# Add core_kernel to path (mock setup)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core_kernel')))

def parse_sec_filing(filing_text: str, config_path: str, prompt_path: str) -> dict:
    """
    Mock function to simulate parsing SEC filings using the provided configuration
    and system prompts, utilizing the core kernel framework.
    """
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return {}

    # Load prompt schema
    try:
        with open(prompt_path, 'r') as f:
            prompt_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_path}")
        return {}

    print(f"Initializing {config.get('domain')} pipeline...")
    print(f"Applying prompt: {prompt_data.get('name')} (v{prompt_data.get('version')})")

    # Mock output matching the schema
    result = {
        "company": "Acme Corp",
        "cik": "0001234567",
        "debt_covenants": [
            {
                "covenant_type": "Leverage Ratio",
                "threshold": 4.5,
                "description": "Total Debt to EBITDA must not exceed 4.5x"
            }
        ],
        "macro_indicators": [
            "Interest rate sensitivity high"
        ]
    }

    return result

if __name__ == "__main__":
    # Example usage
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file = os.path.join(base_dir, 'domains', 'enterprise', 'config.json')
    prompt_file = os.path.join(base_dir, 'prompts', 'enterprise_credit_prompt.json')

    sample_text = "MOCK SEC FILING 10-K..."
    output = parse_sec_filing(sample_text, config_file, prompt_file)
    print("Extraction Result:")
    print(json.dumps(output, indent=2))
